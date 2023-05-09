"""Simple part model."""

from __future__ import annotations

import logging
from argparse import Namespace

import torch

from part_model.dataloader.util import get_metadata
from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

logger = logging.getLogger(__name__)


class SimpleModel(SegClassifier):
    """Simple part model."""

    def __init__(self, args: Namespace, **kwargs):
        """Initialize Simple part model."""
        logger.info("=> Initializing SimpleModel...")
        super().__init__(args, **kwargs)
        exp_tokens = args.experiment.split("-")
        self._use_mask = not self.is_detectron or "mask" in exp_tokens
        self._query_aggregate_mode = "mean"
        self._topk = 1
        if "max" in exp_tokens:
            self._query_aggregate_mode = "max"
        if "top" in args.experiment:
            self._query_aggregate_mode = "topk"
            self._topk = int(
                [t for t in exp_tokens if t.startswith("top")][0][3:]
            )
        metadata = get_metadata(args)

        self._learn_mask = "learn_mask" in exp_tokens
        part_to_class = metadata.part_to_class.clone()
        if self._learn_mask:
            self.part_to_class_mat = torch.nn.Parameter(part_to_class[1:, 1:])
        else:
            self.register_buffer(
                "part_to_class_mat", part_to_class[1:, 1:], persistent=False
            )
        self._num_seg_labels = part_to_class.shape[0]
        
    def _aggregate_queries(self, query_logits: torch.Tensor):
        if self._query_aggregate_mode == "mean":
            return query_logits.mean(1)
        if self._query_aggregate_mode == "max":
            return query_logits.max(1).values
        # Mean over top-k queries
        return query_logits.topk(self._topk, dim=1).mean(1)

    def forward(
        self,
        images: BatchImages,
        samples_list: SamplesList | None = None,
        **kwargs,
    ) -> OutputsDict:
        """End-to-end prediction from images.

        Args:
            images: Batch of images: [B, C, H, W].
            samples_list: List of sampples for detectron2 models.

        Returns:
            Output dict containing predicted classes and segmentation masks in
            logit form and optionally losses dict.
        """
        _ = kwargs  # Unused
        return_dict = self._init_return_dict()
        batch_size, _, height, width = images.shape
        seg_mask_shape = (batch_size, self._num_seg_labels, height, width)
        if self._normalize is not None:
            images = (images - self.mean) / self.std

        # Segmentation part
        if self.is_detectron:
            # Keep samples_list in case it is replaced by new samples_list kwarg
            tmp_samples_list = self._forward_args.get("samples_list", None)
            if samples_list is not None:
                self._forward_args["samples_list"] = samples_list
            outputs, losses = self._base_model(images, **self._forward_args)
            logits_masks = torch.stack(
                [o["sem_seg"].to(self.device) for o in outputs], dim=0
            )
            return_dict["losses"] = losses
            # Put back the original samples_list
            self._forward_args["samples_list"] = tmp_samples_list
        else:
            logits_masks = self._base_model(images, **self._forward_args)

        if "query_clf_logits" in outputs[0]:
            # query_clf_logits: [N, num_queries, num_classes]
            query_clf_logits = torch.stack(
                [o["query_clf_logits"] for o in outputs], dim=0
            )
            cls_logits = query_clf_logits.mean(1)
        elif self._use_mask:
            cls_logits_masks = torch.einsum(
                "bphw,pc->bchw", logits_masks[:, :-1], self.part_to_class_mat
            )
            cls_logits = cls_logits_masks.mean((2, 3))
        else:
            # query_cls_logits: [N, num_queries, num_parts (+1)]
            query_cls_logits = torch.stack(
                [o["query_cls_logits"] for o in outputs], dim=0
            )
            if self._seg_include_bg:
                # If background is included in segmentation, remove it here
                query_cls_logits = query_cls_logits[..., :-1]
            # Sum over parts to get class logits per query.
            cls_logits_queries = torch.einsum(
                "bqp,pc->bqc", query_cls_logits, self.part_to_class_mat
            )
            # Average over queries to get class logits
            cls_logits = cls_logits_queries.mean(1)

        assert logits_masks.shape == seg_mask_shape, (
            f"Shape of seg masks {logits_masks.shape} must be equal to "
            f"{seg_mask_shape}!"
        )
        assert cls_logits.shape == (batch_size, self._num_classes), (
            f"cls_logits.shape {cls_logits.shape} must be equal to "
            f"{(batch_size, self._num_classes)}!"
        )
        return_dict["class_logits"] = cls_logits
        return_dict["sem_seg_logits"] = logits_masks
        return_dict["sem_seg_masks"] = logits_masks.detach().argmax(1)
        return return_dict
