"""Weighted bounding-box part model."""

from __future__ import annotations

import logging
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from part_model.dataloader.util import get_metadata
from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

_EPS = 1e-6

logger = logging.getLogger(__name__)


class WeightedBBoxFeatureExtractor(nn.Module):
    """Feature extraction layer for WeightedBBox part model."""

    def __init__(
        self,
        height: int,
        width: int,
        norm_by_img: bool,
        no_score: bool,
        use_conv1d: bool,
    ) -> None:
        """Initialize WeightedBBoxFeatureExtractor.

        Args:
            height: _description_
            width: _description_
            norm_by_img: _description_
            no_score: _description_
            use_conv1d: _description_
        """
        super().__init__()
        self.height = height
        self.width = width
        self.norm_by_img = norm_by_img
        self.no_score = no_score
        self.use_conv1d = use_conv1d
        grid = torch.arange(height)[None, None, :]
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, logits_masks: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract features."""
        _ = kwargs  # Unused
        # masks: [B, num_segs (including background), H, W]
        masks = F.softmax(logits_masks, dim=1)
        # Remove background
        masks = masks[:, 1:]

        # Compute foreground/background mask (fg_score - bg_score)
        fg_mask = (
            logits_masks[:, 1:].sum(1, keepdim=True) - logits_masks[:, 0:1]
        )
        fg_mask = torch.sigmoid(fg_mask)
        fg_mask = fg_mask / fg_mask.sum((2, 3), keepdim=True).clamp_min(_EPS)
        # weighted_logits_masks = logits_masks[:, 1:] * fg_mask
        # masks = F.softmax(weighted_logits_masks, dim=1)

        # out: [batch_size, num_classes]
        class_scores = (logits_masks[:, 1:] * fg_mask).sum((2, 3))

        # Compute mean and sd for part mask
        mask_sums = torch.sum(masks, [2, 3]) + _EPS
        mask_sums_x = torch.sum(masks, 2) + _EPS
        mask_sums_y = torch.sum(masks, 3) + _EPS

        # Part centroid is standardized by object's centroid and sd
        x_center = (mask_sums_x * self.grid).sum(2) / mask_sums
        y_center = (mask_sums_y * self.grid).sum(2) / mask_sums
        x_std = (mask_sums_x * (self.grid - x_center.unsqueeze(-1)) ** 2).sum(
            2
        ) / mask_sums
        y_std = (mask_sums_y * (self.grid - y_center.unsqueeze(-1)) ** 2).sum(
            2
        ) / mask_sums
        x_std = x_std.sqrt()
        y_std = y_std.sqrt()

        if self.norm_by_img:
            # Normalize centers to [-1, 1]
            x_center = x_center / self.width * 2 - 1
            y_center = y_center / self.height * 2 - 1
            # Max sdX is W / 2 (two pixels on 0 and W-1). Normalize to [0, 1]
            x_std = x_std / self.width * 2
            y_std = y_std / self.height * 2

        if self.no_score:
            centroids = [x_center, y_center, x_std, y_std]
        else:
            centroids = [class_scores, x_center, y_center, x_std, y_std]
        # segOut: [batch_size, num_classes/parts, num_features (4 or 5)]
        centroids = torch.cat([s.unsqueeze(-1) for s in centroids], dim=2)
        if self.use_conv1d:
            centroids = centroids.permute(0, 2, 1)
        return centroids


class WeightedBBoxModel(SegClassifier):
    """Weighted bounding-box part model."""

    def __init__(self, args: Namespace, **kwargs) -> None:
        """Initialize WeightedBBoxModel."""
        super().__init__(args, **kwargs)
        logger.info("=> Initializing WeightedBBoxModel...")
        use_conv1d = "conv1d" in args.experiment
        no_score = "no_score" in args.experiment
        norm_by_img = "norm_img" in args.experiment
        self.return_centroid = "centroid" in args.experiment

        dim = 4 if no_score else 5
        dim_per_bbox = 10 if use_conv1d else dim
        input_dim = (args.seg_labels - 1) * dim_per_bbox
        metadata = get_metadata(args)
        bg_idx = 1
        self.register_buffer(
            "part_to_class_mat",
            metadata.part_to_class[bg_idx:, bg_idx:][None, :, :, None, None],
            persistent=False,
        )
        _, height, width = metadata.input_dim
        self._total_pixels = height * width

        self.core_model = nn.Sequential(
            nn.Conv1d(dim, 10, 1) if use_conv1d else nn.Identity(),
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, args.num_classes),
        )

        grid = torch.arange(height)[None, None, :]
        self.register_buffer("grid", grid, persistent=False)
        self.feature_extactor = WeightedBBoxFeatureExtractor(
            height, width, norm_by_img, no_score, use_conv1d
        )

    def get_classifier(self) -> nn.Module:
        """Returns the classifier part of the model."""
        return nn.Sequential(self.feature_extactor, self.core_model)

    def forward(
        self,
        images: BatchImages,
        samples_list: SamplesList | None = None,
        **kwargs,
    ) -> OutputsDict:
        """See SegClassifier.forward for details."""
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

        bbox_features = self.feature_extactor(logits_masks)
        class_logits = self.core_model(bbox_features)
        assert class_logits.shape == (batch_size, self._num_classes)

        return_dict["class_logits"] = class_logits
        return_dict["sem_seg_logits"] = logits_masks
        return_dict["sem_seg_masks"] = logits_masks.detach().argmax(1)

        if self.return_centroid:
            # Get softmax mask and remove background
            bbox_features = F.softmax(logits_masks, dim=1)
            bbox_features = bbox_features[:, 1:]
            object_masks = bbox_features.unsqueeze(2) * self.part_to_class_mat
            object_masks = object_masks.sum(1)
            object_masks_sums = object_masks.sum((2, 3)) / self._total_pixels
            x_center = bbox_features[:, :, -4]
            y_center = bbox_features[:, :, -3]
            centroids = (x_center, y_center, object_masks_sums)
            return_dict["centroids"] = centroids

        return return_dict
