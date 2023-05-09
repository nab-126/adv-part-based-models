"""Segmentation models."""

from __future__ import annotations

import logging
from collections import OrderedDict

import segmentation_models_pytorch as smp
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from part_model.dataloader.util import get_metadata
from part_model.models.common import Normalize
from part_model.utils.distributed import is_dist_avail_and_initialized

# TODO(feature): Add more backbones
_WEIGHT_URLS = {
    "maskdino": {
        "imagenet": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        "cityscapes": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        "ade20k": "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
    },
    "mask2former": {
        "imagenet": "detectron2://ImageNetPretrained/torchvision/R-50.pkl",
        "cityscapes": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl",
        "ade20k": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_R50_bs16_160k/model_final_500878.pkl",
        "mapillary": "https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer_R50_bs16_300k/model_final_6c66d0.pkl",
    },
}

logger = logging.getLogger(__name__)


def build_deeplabv3(args, normalize: bool = True):
    """Build DeepLabv3 model.

    Args:
        args: Arguments.
        normalize: If True, normalize inputs. Defaults to True.

    Returns:
        DeepLabv3 model.
    """
    # NOTE: DeepLabV3 is pretrained on COCO (not ImageNet)
    model = torch.hub.load(
        "pytorch/vision:v0.10.0",
        "deeplabv3_resnet50",
        pretrained=args.pretrained,
    )
    model.classifier = DeepLabHead(2048, args.seg_labels)
    model.aux_classifier = None

    if normalize:
        normalize = get_metadata(args.dataset)["normalize"]
        model = nn.Sequential(Normalize(**normalize), model)

    if args.seg_dir != "":
        best_path = f"{args.seg_dir}/checkpoint_best.pt"
        print(f"=> loading best checkpoint for DeepLabv3: {best_path}")
        if args.gpu is None:
            checkpoint = torch.load(best_path)
        else:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(best_path, map_location=f"cuda:{args.gpu}")

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]  # remove `module`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model


def build_deeplabv3plus(args, normalize: bool = True):
    """Build DeepLabv3+ model.

    Args:
        args: Arguments.
        normalize: If True, normalize inputs. Defaults to True.

    Returns:
        DeepLabv3+ model.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=args.seg_backbone,
        encoder_weights=args.pretrained,
        in_channels=3,
        classes=args.seg_labels,
        # Default parameters
        encoder_depth=5,
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        upsampling=4,
        aux_params=None,
    )
    if normalize:
        normalize = get_metadata(args.dataset)["normalize"]
        model = nn.Sequential(Normalize(**normalize), model)

    if args.seg_dir != "":
        best_path = f"{args.seg_dir}/checkpoint_best.pt"
        print(f"=> loading best checkpoint for DeepLabv3+: {best_path}")
        if args.gpu is None:
            checkpoint = torch.load(best_path)
        else:
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(best_path, map_location=f"cuda:{args.gpu}")

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k[7:]  # remove `module`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    return model


def build_detectron2_models(args, normalize: bool = True) -> nn.Module:
    """Build a detectron2 model.

    # TODO(enhancement): We may want to make a model-specific build function if
    # it gets too complicated.
    """
    # Register our custom models. pylint: disable=unused-import,import-outside-toplevel
    import part_model.models.mask2former.maskformer_model
    import part_model.models.maskdino.maskdino

    cfg = args.cfg
    exp_tokens = args.experiment.split("-")
    metadata = MetadataCatalog.get(f"{args.dataset}_train")
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = metadata.model_num_classes
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = metadata.model_ignore_label
    cfg.MODEL.RESNETS.NORM = (
        "SyncBN" if is_dist_avail_and_initialized() else "BN"
    )

    # Decoder params
    if args.seg_arch == "maskdino":
        logger.info("Building MaskDINO model...")
        cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "CustomMaskDINODecoder"
        valid_tokens = ["clf", "clf2", "clf3", "default"]
        mode_token = [t for t in exp_tokens if t in valid_tokens]
        decoder_mode = mode_token[0] if mode_token else "default"
        clf_head_dim = args.num_classes + 1
        hl_token = [t for t in exp_tokens if "hl" in t]
        clf_head_layers = int(hl_token[0][2:]) if hl_token else 1
        cfg.MODEL.MaskDINO.decoder_mode = decoder_mode
        cfg.MODEL.MaskDINO.clf_head_dim = clf_head_dim
        cfg.MODEL.MaskDINO.clf_head_layers = clf_head_layers
        cfg.MODEL.MaskDINO.COST_CLF_WEIGHT = 0.0
        cfg.MODEL.MaskDINO.CLF_WEIGHT = 0.0
        logger.info("Using MaskDINO decoder mode: %s", decoder_mode)

        if decoder_mode != "default":
            logger.info(
                "clf_head_dim=%d, clf_head_layers=%d",
                clf_head_dim,
                clf_head_layers,
            )
            if "clf" in decoder_mode:
                cfg.MODEL.MaskDINO.COST_CLF_WEIGHT = (
                    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
                )
                cfg.MODEL.MaskDINO.CLF_WEIGHT = cfg.MODEL.MaskDINO.CLASS_WEIGHT
    elif args.seg_arch == "mask2former":
        logger.info("Building Mask2Former model...")

    if normalize:
        normalize = get_metadata(args.dataset).normalize
        # NOTE: this normalization is not really used since we normalize by the
        # wrapper, Classifier or SegClassifier.
        cfg.MODEL.PIXEL_MEAN = normalize["mean"]
        cfg.MODEL.PIXEL_STD = normalize["std"]

    # cfg for building ResNet in detectron2 (should not need edit)
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py#L614
    # norm = cfg.MODEL.RESNETS.NORM
    # stem = BasicStem(
    #     in_channels=input_shape.channels,
    #     out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
    #     norm=norm,
    # )
    # freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    # out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    # depth               = cfg.MODEL.RESNETS.DEPTH
    # num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    # width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    # bottleneck_channels = num_groups * width_per_group
    # in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    # out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    # stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    # res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    # deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    # deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    # deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS

    model = build_model(cfg)
    if args.pretrained is not None:
        DetectionCheckpointer(model).load(
            _WEIGHT_URLS[args.seg_arch][args.pretrained]
        )
    return model


SEGM_BUILDER = {
    "deeplabv3": build_deeplabv3,
    "deeplabv3plus": build_deeplabv3plus,
    "maskdino": build_detectron2_models,
    "mask2former": build_detectron2_models,
}
