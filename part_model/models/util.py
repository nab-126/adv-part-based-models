"""Utility functions for general models."""

from __future__ import annotations

import logging
import os
from argparse import Namespace

import timm
import torch
from torch import nn
from torch.cuda import amp

from part_model.dataloader.util import get_metadata
from part_model.models.base_classifier import Classifier
from part_model.models.det_part_models import (
    dino_bbox_model,
    multi_head_dino_bbox_model,
)
from part_model.models.seg_part_models import (
    clean_mask_model,
    groundtruth_mask_model,
    part_concat_model,
    part_fc_model,
    part_mask_model,
    pixel_count_model,
    pooling_model,
    simple_model,
    two_head_model,
    weighted_bbox_model,
)
from part_model.models.seg_part_models.util import SEGM_BUILDER
from part_model.utils.image import get_seg_type

DETECTRON2_MODELS = {"maskdino", "mask2former"}
PART_MODEL_INIT = {
    "mask": part_mask_model.PartMaskModel,
    "clean": clean_mask_model.CleanMaskModel,
    "groundtruth": groundtruth_mask_model.GroundtruthMaskModel,
    "pooling": pooling_model.PoolingModel,
    "sim": simple_model.SimpleModel,
    "cat": part_concat_model.PartConcatModel,
    "2heads_e": two_head_model.TwoHeadModel,
    "2heads_d": two_head_model.TwoHeadModel,
}

logger = logging.getLogger(__name__)


def wrap_distributed(args: Namespace, model: Classifier) -> nn.Module:
    """Wrap model for distributed training."""
    # NOTE: When using efficientnet as backbone, pytorch's torchrun complains
    # about unused parameters. This can be suppressed by setting
    # find_unused_parameters to True.
    find_unused_parameters: bool = False
    if (args.arch is not None and "efficientnet" in args.arch) or (
        args.seg_backbone is not None and "efficientnet" in args.seg_backbone
    ):
        find_unused_parameters = True

    if args.distributed:
        model.cuda(args.gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model.cuda()
        model = torch.nn.parallel.DataParallel(model)
    return model


def replace_first_layer(model: nn.Module, new_num_channels: int) -> None:
    """Replace first layer of model with Conv2d layer with new_num_channels."""
    last_layer_name = None
    for i, (name, _) in enumerate(model.named_children()):
        if i == 0:
            first_layer_name = name
        last_layer_name = name
    assert last_layer_name is not None, "classifier has no layers!"
    assert (
        first_layer_name != last_layer_name
    ), "classifier must have at least 2 layers!"
    logger.info(
        "First layer of the model defined by `arch` will be replaced to match "
        "with the new input feature dimension. Please make sure that this "
        "module name '%s' is indeed the first layer.",
        first_layer_name,
    )

    # Change first layer of classifier to accept concatenated masks
    setattr(
        model,
        first_layer_name,
        nn.Conv2d(
            new_num_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        ),
    )


def load_checkpoint(
    args: Namespace,
    model: nn.Module,
    model_path: str | None = None,
    resume_opt_state: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
) -> None:
    """Load checkpoint from file.

    Args:
        args: Arguments.
        model: The model to load checkpoint.
        model_path: Path to saved checkpoint. Defaults to None.
        resume_opt_state: If True, also resume optimizer and scaler states.
            Defaults to True.
        optimizer: Optimizer to resume state. Defaults to None.
        scaler: Scaler to resume state. Defaults to None.
    """
    logger.info("=> Loading resume checkpoint %s...", model_path)
    if args.gpu is None:
        checkpoint = torch.load(model_path)
    else:
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(model_path, map_location=f"cuda:{args.gpu}")

    if args.load_from_segmenter:
        logger.info("=> Loading segmenter weight only...")
        state_dict = {}
        for name, params in checkpoint["state_dict"].items():
            name.replace("module", "module.segmenter")
            state_dict[name] = params
        model.load_state_dict(state_dict, strict=False)
    else:
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        except RuntimeError as err:
            logger.warning(str(err))
            logger.warning(
                "Failed to load checkpoint with strict=True. Trying to load "
                "with strict=False..."
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)

    if not args.load_weight_only or resume_opt_state:
        args.start_epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info("=> Loaded resume checkpoint (epoch %d)", checkpoint["epoch"])


def build_classifier(args):
    """Build classifier model.

    Args:
        args: Arguments.

    Returns:
        model, optimizer, scaler
    """
    metadata = get_metadata(args)
    if metadata is None:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}!")
    args.is_detectron = args.seg_arch in DETECTRON2_MODELS

    normalize = metadata.normalize
    model: nn.Module | None = None
    if args.arch is not None:
        # Build a normal classifier
        model = timm.create_model(
            args.arch, pretrained=args.pretrained is not None, num_classes=0
        )
        with torch.no_grad():
            dummy_input = torch.zeros((2,) + metadata.input_dim)
            rep_dim = model(dummy_input).size(-1)

    if get_seg_type(args) is not None:
        # Build segmentation-based models
        tokens = args.experiment.split("-")
        model_token = tokens[1]
        exp_tokens = tokens[2:]

        if args.seg_arch is not None:
            logger.info("=> Building segmentation model...")
            segmenter = SEGM_BUILDER[args.seg_arch](args, normalize=False)

        if args.freeze_seg:
            # Froze all weights of the part segmentation model
            for param in segmenter.parameters():
                param.requires_grad = False

        num_channels: int = (
            args.seg_labels
            + (3 if "inpt" in exp_tokens else 0)
            - (1 if "nobg" in exp_tokens else 0)
        )

        if model_token in ("mask", "clean", "groundtruth"):
            # DEPRECATED: Not tested with new interface. For new interface,
            # see "cat" model below (replace_first_layer is called in init).
            replace_first_layer(model, num_channels)
            model.fc = nn.Linear(rep_dim, args.num_classes)
            model = PART_MODEL_INIT[model_token](args, segmenter, model)
        elif model_token == "cat":
            model = PART_MODEL_INIT[model_token](
                args,
                base_model=segmenter,
                normalize=normalize,
                classifier=model,
                rep_dim=rep_dim,
            )
        elif model_token == "pixel":
            # DEPRECATED: Not tested with new interface.
            model = pixel_count_model.PixelCountModel(args, segmenter, None)
        elif model_token == "bbox_2heads_d":
            # DEPRECATED: Not tested with new interface.
            model = multi_head_dino_bbox_model.MultiHeadDinoBoundingBoxModel(
                args
            )
        elif model_token == "bbox":
            # two options, either bbox model from object detection or bbox from
            # segmentation model
            assert args.obj_det_arch == "dino"
            # DEPRECATED: Not tested with new interface.
            model = dino_bbox_model.DinoBoundingBoxModel(args)
        elif model_token == "wbbox":
            model = weighted_bbox_model.WeightedBBoxModel(
                args, base_model=segmenter, normalize=normalize
            )
        elif model_token == "fc":
            # DEPRECATED: Not tested with new interface.
            model = part_fc_model.PartFCModel(args, segmenter)
        else:
            model = PART_MODEL_INIT[model_token](
                args, base_model=segmenter, normalize=normalize
            )

        n_seg = sum(p.numel() for p in model.parameters()) / 1e6
        nt_seg = (
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        )
        logger.info("=> Model params (train/total): %.2fM/%.2fM", nt_seg, n_seg)
    else:
        logger.info("=> Building a normal classifier...")
        model.fc = nn.Linear(rep_dim, args.num_classes)
        model = Classifier(args, base_model=model, normalize=normalize)
        n_model = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("=> Total params: %.2fM", n_model)

    # Wrap model again under DistributedDataParallel or just DataParallel
    model = wrap_distributed(args, model)

    if args.obj_det_arch == "dino" or model.module.is_detectron:
        logger.info("Setting up DINO/Detectron2 optimizer...")
        backbone_params, non_backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                non_backbone_params.append(param)
        optim_params = [
            {
                "name": "non_backbone",
                "params": non_backbone_params,
                "lr": args.lr,
            },
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": args.lr_backbone
                if args.lr_backbone is not None
                else args.lr,
            },
        ]
    else:
        logger.info("Setting up basic optimizer...")
        p_wd, p_non_wd = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(kword in name for kword in ["bias", "ln", "bn"]):
                p_non_wd.append(param)
            else:
                p_wd.append(param)
        optim_params = [
            {"params": p_wd, "weight_decay": args.wd},
            {"params": p_non_wd, "weight_decay": 0},
        ]

    if args.optim == "sgd":
        logger.info("Using SGD optimizer")
        optimizer = torch.optim.SGD(
            optim_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        logger.info("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            optim_params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.wd,
        )

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # Optionally resume from a checkpoint
    if not (args.evaluate or args.resume or args.resume_if_exist):
        logger.info("=> No model checkpoint resumed.")
        return model, optimizer, scaler

    if args.evaluate:
        model_path = f"{args.output_dir}/checkpoint_best.pt"
        resume_opt_state = False
    else:
        model_path = f"{args.output_dir}/checkpoint_last.pt"
        resume_opt_state = True
        if not args.resume_if_exist or not os.path.isfile(model_path):
            model_path = args.resume
            resume_opt_state = False

    if os.path.isfile(model_path):
        load_checkpoint(
            args,
            model,
            model_path=model_path,
            resume_opt_state=resume_opt_state,
            optimizer=optimizer,
            scaler=scaler,
        )
    elif args.resume:
        raise FileNotFoundError(f"=> No checkpoint found at {model_path}.")
    else:
        logger.info(
            "=> resume_if_exist is True, but no checkpoint found at %s",
            model_path,
        )

    return model, optimizer, scaler


def build_segmentation(args):
    """Build segmentation model.

    Args:
        args: Command line arguments.

    Returns:
        model, optimizer, scaler
    """
    model = SEGM_BUILDER[args.seg_arch](args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = wrap_distributed(args, model)
    model_without_ddp = model.module[1]

    backbone_params = list(model_without_ddp.encoder.parameters())
    last_params = list(model_without_ddp.decoder.parameters())
    last_params.extend(list(model_without_ddp.segmentation_head.parameters()))
    optimizer = torch.optim.SGD(
        [
            {"params": filter(lambda p: p.requires_grad, backbone_params)},
            {"params": filter(lambda p: p.requires_grad, last_params)},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # Optionally resume from a checkpoint
    if args.resume and not args.evaluate:
        if os.path.isfile(args.resume):
            logger.info("=> loading resume checkpoint %s...", args.resume)
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{args.gpu}"
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])

            if not args.load_weight_only:
                args.start_epoch = checkpoint["epoch"]
                optimizer.load_state_dict(checkpoint["optimizer"])
                scaler.load_state_dict(checkpoint["scaler"])
            logger.info(
                "=> loaded resume checkpoint (epoch %d)", checkpoint["epoch"]
            )
        else:
            logger.info("=> no checkpoint found at %s", args.resume)

    return model, optimizer, scaler


def build_model(args):
    """Build model.

    Args:
        args: Command line arguments.

    Returns:
        model, optimizer, scaler
    """
    if "seg-only" in args.experiment:
        return build_segmentation(args)
    return build_classifier(args)
