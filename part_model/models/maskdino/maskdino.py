"""Mask DINO (modified from MaskDINO by IDEA Research)."""

import logging
from typing import Tuple

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from torch import nn
from torch.nn import functional as F

# Register CustomMaskDINODecoder. pylint: disable=unused-import
import part_model.models.maskdino.maskdino_decoder
from MaskDINO.maskdino.utils import box_ops
from part_model.dataloader.util import get_metadata
from part_model.models.maskdino.criterion import CustomSetCriterion
from part_model.models.maskdino.matcher import HungarianMatcher
from part_model.utils.types import (
    BatchImages,
    DetectronOutputsList,
    SamplesList,
)

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MaskDINOCustom(nn.Module):
    """Main class for mask classification semantic segmentation architecture."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        # Custom parameters
        decoder_mode: str = "default",
        dataset: str = "coco",
    ) -> None:
        """Initialize MaskDINO.

        Args:
            backbone: Backbone module, must follow detectron2's interface.
            sem_seg_head: Module that predicts semantic segmentation from
                backbone features.
            criterion: Module that defines the loss.
            num_queries: Number of queries.
            object_mask_threshold: Threshold to filter query based on
                classification score for panoptic segmentation inference.
            overlap_threshold: Overlap threshold used in general inference for
                panoptic segmentation.
            metadata: dataset meta, get `thing` and `stuff` category names for
                panoptic segmentation inference.
            size_divisibility: Some backbones require the input height and width
                to be divisible by a specific integer. Use this to override such
                requirement.
            sem_seg_postprocess_before_inference: Whether to resize prediction
                back to original input size before semantic segmentation
                inference or after. For high-resolution dataset like Mapillary,
                resizing predictions before inference will cause OOM error.
            pixel_mean: Per-channel mean for normalizing the input image.
            pixel_std: Per-channel std for normalizing the input image.
            semantic_on: Whether to output semantic segmentation prediction.
            instance_on: Whether to output instance segmentation prediction.
            panoptic_on: Whether to output panoptic segmentation prediction.
            test_topk_per_image: Instance segmentation parameter, keep topk
                instances per image.
            data_loader: Data loader name (include "detr" or not).
            pano_temp: Temperature for panoptic segmentation inference.
            focus_on_box: Whether to focus on the box region.
            transform_eval: Transform sigmoid score into softmax score to make
                score sharper.
            semantic_ce_loss: Whether use cross-entroy loss in classification.
            decoder_mode: Mode for decoder: "default" (default), "clf" (add
                classification head).
            dataset: Dataset name.
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = (
            sem_seg_postprocess_before_inference
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False
        )

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss
        self._decoder_mode = decoder_mode

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        logger.info(
            "SetCriterion weight_dict: %s", str(self.criterion.weight_dict)
        )

        # (2) For meta-part labels
        # fg_embed: [Q, F], class: [Q, M], mask: [Q, H, W]
        # (2.1) Use fg_embed: [Q, F], fg_to_parts: [F, P] -> class: [Q, P]
        # compute loss for meta-parts.
        # (2.2) Use fg_embed: [Q, F], fg_to_fg-parts: [F, FP] -> fg-part score:
        # [Q, FP]. Then compute loss for fg-parts.
        local_metadata = get_metadata(dataset)
        part_to_class = local_metadata.part_to_class.clone()
        part_to_class = part_to_class.roll(shifts=-1, dims=0)
        # Shape [FP, C]
        self.register_buffer(
            "part_to_class_mat", part_to_class[:, 1:], persistent=False
        )
        if hasattr(local_metadata, "part_to_metapart_mat"):
            part_to_metapart = local_metadata.part_to_metapart_mat.clone()
            part_to_metapart = part_to_metapart.roll(
                shifts=(-1, -1), dims=(0, 1)
            )
            self.register_buffer(
                "part_to_metapart_mat", part_to_metapart, persistent=False
            )

    @classmethod
    def from_config(cls, cfg):
        """Init from config."""
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Building matcher
        matcher = HungarianMatcher(
            cost_class=cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT,
            cost_mask=cfg.MODEL.MaskDINO.COST_MASK_WEIGHT,
            cost_dice=cfg.MODEL.MaskDINO.COST_DICE_WEIGHT,
            cost_box=cfg.MODEL.MaskDINO.COST_BOX_WEIGHT,
            cost_giou=cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT,
            cost_clf=cfg.MODEL.MaskDINO.COST_CLF_WEIGHT,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": cfg.MODEL.MaskDINO.CLASS_WEIGHT,
            "loss_mask": cfg.MODEL.MaskDINO.MASK_WEIGHT,
            "loss_dice": cfg.MODEL.MaskDINO.DICE_WEIGHT,
            "loss_bbox": cfg.MODEL.MaskDINO.BOX_WEIGHT,
            "loss_giou": cfg.MODEL.MaskDINO.GIOU_WEIGHT,
        }
        if "clf" in cfg.MODEL.MaskDINO.decoder_mode:
            weight_dict.update({"loss_clf": cfg.MODEL.MaskDINO.CLF_WEIGHT})

        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update(
                {f"{k}_interm": v for k, v in weight_dict.items()}
            )
            weight_dict.update(interm_weight_dict)

        # denoising training
        dn_losses = []
        if cfg.MODEL.MaskDINO.DN == "standard":
            weight_dict.update(
                {
                    f"{k}_dn": v
                    for k, v in weight_dict.items()
                    if k not in ("loss_mask", "loss_dice")
                }
            )
            dn_losses = ["labels", "boxes"]
        elif cfg.MODEL.MaskDINO.DN == "seg":
            weight_dict.update({f"{k}_dn": v for k, v in weight_dict.items()})
            dn_losses = ["labels", "masks", "boxes"]

        losses = ["labels", "masks"]
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses.append("boxes")

        if "clf" in cfg.MODEL.MaskDINO.decoder_mode:
            losses.append("clf")
            dn_losses.append("clf")

        if cfg.MODEL.MaskDINO.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()}
                )
            weight_dict.update(aux_weight_dict)

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # building criterion
        criterion = CustomSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
            and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS
            and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            num_clf_classes=cfg.MODEL.MaskDINO.clf_head_dim - 1,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": metadata,
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": (
                cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON
                and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS
                and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
            ),
            # Custom parameters
            "decoder_mode": cfg.MODEL.MaskDINO.decoder_mode,
            "dataset": cfg.DATASETS.TRAIN[0].replace("_train", ""),
        }

    @property
    def device(self):
        """Hack to get the device of the model."""
        return self.pixel_mean.device

    def forward(
        self, images: BatchImages, samples_list: SamplesList | None = None
    ) -> tuple[DetectronOutputsList, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            images: A batch of images. Tensor of shape (N, C, H, W).
            samples_list: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                    * "image": Tensor, image in (C, H, W) format.
                    * "instances": per-region ground truth
                    * Other information that's included in the original dicts,
                      such as: "height", "width" (int): the output resolution of
                      the model (may be different from input resolution), used
                      in inference.

        Returns:
            list[dict]: each dict has the results for one image. The dict
                contains the following keys:
                * "sem_seg": A Tensor that represents the per-pixel segmentation
                    prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg": A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the
                    values are ids for each segment. segments_info (list[dict]):
                    Describe each segment in `panoptic_seg`. Each dict contains
                    keys "id", "category_id", "isthing".
        """
        batch_size = images.shape[0]
        assert batch_size == len(samples_list), (
            f"Batch size {batch_size} does not match the number of samples "
            f"{len(samples_list)} in samples_list"
        )
        image_sizes = [img.shape[-2:] for img in images]
        features = self.backbone(images)

        if self.training:
            assert (
                samples_list is not None
            ), "samples_list must be provided in training mode."

            # mask classification target
            targets = None
            if "instances" in samples_list[0]:
                gt_instances = [
                    x["instances"].to(self.device) for x in samples_list
                ]
                if "detr" in self.data_loader:
                    targets = self.prepare_targets_detr(gt_instances, images)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            outputs, mask_dict = self.sem_seg_head(features, targets=targets)

            # Pop None outputs
            empty_keys = [key for key, val in outputs.items() if val is None]
            for key in empty_keys:
                outputs.pop(key)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_dict)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
        else:
            losses = {}
            outputs, _ = self.sem_seg_head(features)

        # Upsample masks to image size
        outputs["pred_masks"] = F.interpolate(
            outputs["pred_masks"],
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        outputs_list = [
            outputs.get(name, [None] * batch_size)
            for name in ["pred_logits", "pred_masks", "pred_boxes", "pred_clf"]
        ]
        del outputs
        processed_results = []
        for (
            mask_cls_result,
            mask_pred_result,
            mask_box_result,
            mask_clf_result,
            image_size,
        ) in zip(*outputs_list, image_sizes):
            height, width = image_size
            processed_results.append({})
            new_size = mask_pred_result.shape[-2:]
            # mask_cls_result: shape (num_queries, num_classes)
            processed_results[-1]["query_cls_logits"] = mask_cls_result[:, :-1]

            if mask_clf_result is not None:
                processed_results[-1]["query_clf_logits"] = mask_clf_result[
                    :, :-1
                ]

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result, mask_clf=mask_clf_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                    mask_cls_result, mask_pred_result
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                mask_box_result = mask_box_result.to(mask_pred_result)
                height = new_size[0] / image_size[0] * height
                width = new_size[1] / image_size[1] * width
                mask_box_result = self.box_postprocess(
                    mask_box_result, height, width
                )

                instance_r = retry_if_cuda_oom(self.instance_inference)(
                    mask_cls_result, mask_pred_result, mask_box_result
                )
                processed_results[-1]["instances"] = instance_r

        assert len(processed_results) == batch_size, (
            f"The number of results ({len(processed_results)}) does not match "
            f"the number of inputs ({batch_size})."
        )
        return processed_results, losses

    def prepare_targets(self, targets, images):
        """Prepare targets for computing the loss."""
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(
                        targets_per_image.gt_boxes.tensor
                    )
                    / image_size_xyxy,
                    "obj_class": targets_per_image.obj_class,
                }
            )
        return new_targets

    def prepare_targets_detr(self, targets, images):
        """Prepare targets for DETR."""
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes": box_ops.box_xyxy_to_cxcywh(
                        targets_per_image.gt_boxes.tensor
                    )
                    / image_size_xyxy,
                    "obj_class": targets_per_image.obj_class,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, mask_clf=None):
        """Inference semantic segmentation from mask_cls and mask_pred.

        Args:
            mask_cls: shape (num_queries, num_classes + 1).
            mask_pred: shape (num_queries, H, W).
            mask_clf: shape (num_queries, num_clf_classes).
        """
        if "clf" in self._decoder_mode:
            device = mask_cls.device
            part_to_class_mat = self.part_to_class_mat.to(device)
            part_to_metapart_mat = self.part_to_metapart_mat.to(device)
            if self._decoder_mode == "clf":
                # Part scores from classification head
                mask_clf = mask_clf.softmax(dim=-1)[..., :-1]
                part_scores_clf = torch.einsum(
                    "ql,pl->qp", mask_clf, part_to_class_mat
                )
                # Part scores from semantic class head (meta-parts)
                mask_cls = mask_cls.softmax(dim=-1)[..., :-1]
                part_scores_cls = torch.einsum(
                    "qm,pm->qp", mask_cls, part_to_metapart_mat
                )
                # Combine part scores and get segmentation masks
                mask_part = part_scores_clf
                mask_part *= part_scores_cls

            elif self._decoder_mode == "clf2":
                part_scores_clf = torch.einsum(
                    "ql,pl->qp", mask_clf[..., :-1], part_to_class_mat
                )
                part_scores_cls = torch.einsum(
                    "qm,pm->qp", mask_cls[..., :-1], part_to_metapart_mat
                )
                mask_part = part_scores_clf
                mask_part += part_scores_cls

            elif self._decoder_mode == "clf3":
                part_scores_clf = torch.einsum(
                    "ql,pl->qp", mask_clf[..., :-1], part_to_class_mat
                )
                part_scores_cls = torch.einsum(
                    "qm,pm->qp", mask_cls[..., :-1], part_to_metapart_mat
                )
                mask_part = F.softplus(part_scores_clf)
                mask_part *= F.softplus(part_scores_cls)

            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qp,qhw->phw", mask_part, mask_pred)
            return semseg

        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            # TODO(design): We may want to be careful of this use of softmax and
            # sigmoid for adversarial training.
            # Take out background class
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

        # if use focal loss in training, evaluate with sigmoid. As sigmoid is
        # mainly for detection and not sharp enough for semantic and panoptic
        # segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        mask_cls = mask_cls.sigmoid()
        if self.transform_eval:
            mask_cls = F.softmax(
                mask_cls / self.pano_temp, dim=-1
            )  # already sigmoid
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        """Panoptic segmentation inference."""
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid
        # is mainly for detection and not sharp enough for semantic and panoptic
        # segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(
                mask_cls.sigmoid() / self.pano_temp, dim=-1
            ).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros(
            (h, w), dtype=torch.int32, device=cur_masks.device
        )
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        # take argmax
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = (
                pred_class
                in self.metadata.thing_dataset_id_to_contiguous_id.values()
            )
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= prob).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue

                # merge stuff regions
                if not isthing:
                    if int(pred_class) in stuff_memory_list:
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        continue
                    stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )
        return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        """Postprocess predictions."""
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (
                    lab
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        """Postprocess box height and width."""
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes
