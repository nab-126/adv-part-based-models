"""Concatenated segmentation part-based model.

Segmentation part model that concatenates the segmentation mask with the image
and feeds it to a classifier.
"""

from __future__ import annotations

import logging
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from part_model.dataloader.util import get_metadata
from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

logger = logging.getLogger(__name__)


class PartConcatModel(SegClassifier):
    """PartConcatModel feeds segmentation masks and images to classifier."""

    def __init__(
        self,
        args: Namespace,
        classifier: nn.Module | None = None,
        rep_dim: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize PartConcatModel.

        Args:
            args: Arguments.
            classifier: Final classifier model. PartConcatModel allows defining
                any flexible pre-trained classifier. Defaults to None.
            rep_dim: Output dimension of `classifier`. Defaults to None.
        """
        super().__init__(args, **kwargs)
        logger.info("=> Initializing PartConcatModel...")
        if classifier is None:
            raise ValueError("classifier must be provided!")
        self._classifier = classifier
        self._use_hard_mask = "hard" in args.experiment
        self._group_part_by_class = "group" in args.experiment
        self._no_bg = "nobg" in args.experiment
        self._multiply = "mult" in args.experiment

        # Set number of masks to keep
        self._k = args.seg_labels
        if self._group_part_by_class:
            metadata = get_metadata(args)
            self.register_buffer("part_to_class_mat", metadata.part_to_class)
            # Set k to number of object classes
            self._k = self.part_to_class_mat.size(-1)
        # Remove background dim if needed
        self._k -= self._no_bg
        if "top" in args.experiment:
            # DEPRECATED: this is a bug for now
            self._k = int(args.experiment.split("top")[-1].split("-")[0])
            raise NotImplementedError("topk option is not implemented!")
        self._mask = None
        self._temperature = args.temperature

        if self._multiply:
            # Linear layer aggregates output logits from all parts
            self._linear_dim = rep_dim * self._k
            self.linear = nn.Linear(self._linear_dim, args.num_classes)
        else:
            # Prevent circular import. pylint: disable=import-outside-toplevel
            from part_model.models.util import replace_first_layer

            # Change first layer of classifier to accept concatenated masks
            num_channels = self._k + 3
            replace_first_layer(self._classifier, num_channels)

    def forward(
        self,
        images: BatchImages,
        samples_list: SamplesList | None = None,
        **kwargs,
    ) -> OutputsDict:
        """Forward pass. See SegClassifier.forward for details."""
        _ = kwargs  # Unused
        return_dict = self._init_return_dict()
        if self._normalize is not None:
            images = (images - self.mean) / self.std
        batch_size = images.shape[0]
        images = self._apply_mask(images)

        # Segmentation part
        if self.is_detectron:
            # Keep samples_list in case it is replaced by new samples_list kwarg
            tmp_samples_list = self._forward_args.get("samples_list", None)
            if samples_list is not None:
                self._forward_args["samples_list"] = samples_list
            outputs, losses = self._base_model(images, **self._forward_args)
            logits_masks = torch.stack([o["sem_seg"] for o in outputs], dim=0)
            return_dict["losses"] = losses
            # Put back the original samples_list
            self._forward_args["samples_list"] = tmp_samples_list
        else:
            logits_masks = self._base_model(images, **self._forward_args)

        masks = F.softmax(logits_masks / self._temperature, dim=1)
        bg_idx = 1 if self._no_bg else 0

        if self._group_part_by_class:
            # Sum up parts of the same object class
            masks = masks.unsqueeze(2)
            masks *= self.part_to_class_mat[None, :, :, None, None]
            masks = masks.sum(1)

        if not self._use_hard_mask:
            # Use softmax mask
            masks = masks[:, bg_idx:, :, :]
        else:
            # Use one-hot mask (not differentiable)
            label_masks = F.one_hot(masks.argmax(1), num_classes=self._k)
            label_masks = label_masks[:, :, :, bg_idx:]
            masks = label_masks.permute(0, 3, 1, 2)

        if self._multiply:
            # Multiply image with mask
            masks.unsqueeze_(2)
            images = images.unsqueeze(1)
            images *= masks
            # Flatten part dimension to batch dimension
            images = images.reshape(
                (batch_size * (self._k - bg_idx),) + images.shape[2:]
            )
            class_logits = self._classifier(images)
            class_logits = class_logits.view(batch_size, self._linear_dim)
            class_logits = self.linear(class_logits)
        else:
            # Concatenate image with mask
            images = torch.cat([images, masks], dim=1)
            class_logits = self._classifier(images)

        return_dict["class_logits"] = class_logits
        return_dict["sem_seg_logits"] = logits_masks
        return return_dict

    def set_mask(self, gt_masks: BatchImages) -> None:
        """Set binary mask from ground truth segmentation mask.

        This is to zero-out pixels that correspond to -1 mask.
        """
        self._mask = (gt_masks != -1).float()

    def _apply_mask(self, images: BatchImages) -> BatchImages:
        if self._mask is None:
            return images
        return images * self._mask
