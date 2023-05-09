"""Implement the two-headed part models."""

from __future__ import annotations

import argparse
import logging

import torch
from torch import nn

from part_model.models.seg_part_models.seg_classifier import SegClassifier
from part_model.utils.types import BatchImages, OutputsDict, SamplesList

logger = logging.getLogger(__name__)


class AttributeModel(SegClassifier):
    """Attribute part model."""

    def __init__(self, args: argparse.Namespace, **kwargs) -> None:
        """Initialize the two-headed part model.

        Args:
            args: Arguments.
            segmentor: The base segmentation model.
            mode:
        """
        super().__init__(args, **kwargs)

        self._base_model.segmentation_head = Heads(self._base_model, args.num_classes)

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
            class_logits, attribute_logits = outputs

        # assert logits_masks.shape == seg_mask_shape
        assert class_logits.shape == (batch_size, self._num_classes)
        return_dict["class_logits"] = class_logits
        return_dict["attribute_logits"] = attribute_logits
        # return_dict["sem_seg_logits"] = logits_masks
        # return_dict["sem_seg_masks"] = logits_masks.detach().argmax(1)
        return return_dict
    
    # TODO: preprocess detectron2
    _Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    def _preprocess_torchvision(self, inputs: _Inputs) -> _Inputs:
        """Preprocess batch of tuple of samples for generic torchvision models.

        Args:
            inputs: Tuple of tensors.

        Returns:
            Preprocessed batch of tensors.
        """
        if len(inputs) != 6:
            raise ValueError(f"Expected 6 inputs, got {len(inputs)}!")
        prep_inputs = [t.to(self.device, non_blocking=True) for t in inputs]
        # Normalize images (assume as first input)
        if prep_inputs[0].dtype == torch.uint8:
            prep_inputs[0] = prep_inputs[0].float()
            prep_inputs[0] /= 255
        return prep_inputs[0], prep_inputs[1], prep_inputs[2:]

    def preprocess(self, inputs: _Inputs | SamplesList) -> _Inputs:
        """See Classifier.preprocess."""
        # if self.is_detectron:
        #     return self._preprocess_detectron(inputs)
        return self._preprocess_torchvision(inputs)



class Heads(nn.Module):
    """Multi-heads module."""

    def __init__(self, segmentor: nn.Module, num_classes: int) -> None:
        """Initialize the multi-heads module.

        Args:
            segmentor: The base segmentation model.
            num_classes: Number of classes.
        """
        super().__init__()

        pool_size = 7
        fc_dim = 1024
        
        classification_head = nn.Sequential(
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
        )

        # Attribute head
        attribute_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(256*pool_size*pool_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),
        )
        self._color_head = nn.Linear(fc_dim, 30)
        self._pattern_head = nn.Linear(fc_dim, 11)
        self._material_head = nn.Linear(fc_dim, 14)
        self._reflectance_head = nn.Linear(fc_dim, 4)

        self.heads = nn.ModuleList(
            [
                classification_head,
                attribute_head,
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
        out_classification_head = out[0]
        out_attribute_head = out[1]
        # TODO: add boolean to only return class logits

        # TODO: do not hardcode head names
        color_logits = self._color_head(out_attribute_head)
        pattern_logits = self._pattern_head(out_attribute_head)
        material_logits = self._material_head(out_attribute_head)
        reflectance_logits = self._reflectance_head(out_attribute_head)
        return out_classification_head, (color_logits, pattern_logits, material_logits, reflectance_logits)

        if return_mask:
            return out[1], out[0]
        return out[1]
