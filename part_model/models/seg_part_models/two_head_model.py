"""Implement the two-headed part models."""

from __future__ import annotations

import argparse
import logging

import torch
from torch import nn

from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

logger = logging.getLogger(__name__)


class TwoHeadModel(SegClassifier):
    """Two-headed part model."""

    def __init__(self, args: argparse.Namespace, **kwargs) -> None:
        """Initialize the two-headed part model.

        Args:
            args: Arguments.
            segmentor: The base segmentation model.
            mode:
        """
        super().__init__(args, **kwargs)
        # The mode of two-headed part model. "d" for decoder and "e" for
        # "encoder". Format: "...2heads_MODE...".
        self._mode = args.experiment.split("2heads_")[1][1]
        if self._mode == "d":
            logger.warning(
                'mode "d" in TwoHeadModel is outdated and may not work as '
                "expected."
            )
            self._base_model.segmentation_head = Heads(
                self._base_model.segmentation_head, args.num_classes
            )
            self._base_model[1].segmentation_head.returnMask = True
        else:
            latent_dim = 2048  # TODO(enhancement): depends on backbone
            pool_size = 4
            fc_dim = 64
            self._base_model.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.BatchNorm2d(latent_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(latent_dim, fc_dim, (pool_size, pool_size)),
                nn.BatchNorm2d(fc_dim),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(fc_dim, self._num_classes),
            )

    def forward(
        self,
        images: BatchImages,
        samples_list: SamplesList | None = None,
        **kwargs,
    ) -> OutputsDict:
        """Forward pass. See SegClassifier.forward for details.

        Args:
            images: Input images in [B, C, H, W].
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
            raise NotImplementedError()
            # dectectron2 model returns (mask, losses)
            outputs, losses = self._base_model(
                images, samples_list=samples_list
            )
            logits_masks = torch.stack([o["sem_seg"] for o in outputs], dim=0)
            return_dict["losses"] = losses
        else:
            outputs = self._base_model(images, **self._forward_args)
            logits_masks, class_logits = outputs

        assert logits_masks.shape == seg_mask_shape
        assert class_logits.shape == (batch_size, self._num_classes)
        return_dict["class_logits"] = class_logits
        return_dict["sem_seg_logits"] = logits_masks
        return_dict["sem_seg_masks"] = logits_masks.detach().argmax(1)
        return return_dict


class Heads(nn.Module):
    """Multi-heads module."""

    def __init__(self, segmentor: nn.Module, num_classes: int) -> None:
        """Initialize the multi-heads module.

        Args:
            segmentor: The base segmentation model.
            num_classes: Number of classes.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                segmentor.segmentation_head,
                nn.Sequential(
                    nn.Conv2d(256, 50, (3, 3), (1, 1)),
                    nn.BatchNorm2d(50),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(50, 10, (1, 1), (1, 1)),
                    nn.BatchNorm2d(10),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(1690, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes),
                ),
            ]
        )

    def forward(
        self, inputs: BatchImages, return_mask: bool = False
    ) -> BatchImages:
        """Forward pass.

        Args:
            inputs: Input images in [B, C, H, W].

        Returns:
            The predicted class logits and predicted segmentation mask.
        """
        out = [head(inputs) for head in self.heads]
        if return_mask:
            return out[1], out[0]
        return out[1]
