"""Downsampled (pooling) part model."""

from __future__ import annotations

import logging
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

logger = logging.getLogger(__name__)


class PoolingFeatureExtractor(nn.Module):
    """Feature extraction layer for Downsampled part model."""

    def __init__(self, no_bg: bool) -> None:
        """Initialize PoolingFeatureExtractor.

        Args:
            no_bg: If True, background channel of the mask is dropped.
        """
        super().__init__()
        self.no_bg: bool = no_bg

    def forward(
        self, logits_masks: torch.Tensor, from_logits: bool = True
    ) -> torch.Tensor:
        """Extract features.

        Args:
            logits_masks: Predicted masks to extract features from.
            from_logits: If True, expect logits_masks to be logits. Otherwise,
                expect softmax/probability mask.

        Returns:
            Extracted features.
        """
        # masks: [B, num_segs (including background), H, W]
        if from_logits:
            masks = F.softmax(logits_masks, dim=1)
        else:
            masks = logits_masks
        # Remove background
        if self.no_bg:
            masks = masks[:, 1:]
        return masks


class PoolingModel(SegClassifier):
    """Downsampled (or pooling) part model."""

    def __init__(self, args: Namespace, **kwargs):
        """Initialize Downsampled part model."""
        super().__init__(args, **kwargs)
        logger.info("=> Initializing PoolingModel...")
        self.no_bg = "nobg" in args.experiment
        use_bn_after_pooling = "bn" in args.experiment
        input_dim = args.seg_labels - (1 if self.no_bg else 0)

        idx = args.experiment.find("pooling")
        pool_size = int(args.experiment[idx:].split("-")[1])
        var_per_mask = 5

        batchnorm = []
        if use_bn_after_pooling:
            batchnorm = [nn.BatchNorm2d(input_dim), nn.ReLU(inplace=True)]

        self.core_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            *batchnorm,
            nn.Conv2d(
                input_dim, input_dim * var_per_mask, (pool_size, pool_size)
            ),
            nn.BatchNorm2d(input_dim * var_per_mask),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(input_dim * var_per_mask, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, args.num_classes),
        )
        self.feature_extactor = PoolingFeatureExtractor(self.no_bg)

    def get_classifier(self):
        """Get model that takes logit mask and returns classification output."""
        return nn.Sequential(self.feature_extactor, self.core_model)

    def forward(
        self,
        images: BatchImages,
        samples_list: SamplesList | None = None,
        **kwargs,
    ) -> OutputsDict:
        """Forward pass. See SegClassifier.forward for details.

        Args:
            images: Batch of images with shape [B, C, H, W].
            samples_list: List of samples for detectron2 models.

        Returns:
            Output dictionary.
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
            logits_masks = torch.stack([o["sem_seg"] for o in outputs], dim=0)
            return_dict["losses"] = losses
            # Put back the original samples_list
            self._forward_args["samples_list"] = tmp_samples_list
        else:
            logits_masks = self._base_model(images, **self._forward_args)
        assert logits_masks.shape == seg_mask_shape

        masks = self.feature_extactor(logits_masks)
        class_logits = self.core_model(masks)
        assert class_logits.shape == (batch_size, self._num_classes)

        return_dict["class_logits"] = class_logits
        return_dict["sem_seg_logits"] = logits_masks
        return_dict["sem_seg_masks"] = logits_masks.detach().argmax(1)
        return return_dict
