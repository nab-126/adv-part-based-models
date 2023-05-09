"""Base classifier interface."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, List, Tuple, Union

import torch
from torch import nn

from part_model.utils.types import BatchImages, OutputsDict

_NormVal = Union[List[float], Tuple[float, float, float]]
_Inputs = tuple[torch.Tensor, torch.Tensor]


class Classifier(nn.Module):
    """Base Classifier interface."""

    def __init__(
        self,
        args: Namespace,
        base_model: nn.Module | None = None,
        normalize: dict[str, _NormVal] | None = None,
        forward_args: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Classifier.

        Args:
            args: Arguments.
            base_model: Main PyTorch model.
            normalize: Dictionary containing normalization values; must contain
                "mean" and "std". Defaults to None.
            forward_args: Keyword arguments to pass with all forward calls to
                base_model. Can be update via the update_forward_args().
        """
        super().__init__()
        self.is_detectron: bool = args.is_detectron
        # Forward args can be updated via the update_forward_args() method and
        # will be passed to the forward() method of the base model.
        self._forward_args: dict[str, Any] = forward_args or {}
        self._base_model: nn.Module = base_model
        self._normalize: dict[str, _NormVal] | None = normalize
        self._num_classes: int = args.num_classes
        if normalize is not None:
            mean = normalize["mean"]
            std = normalize["std"]
            self.register_buffer(
                "mean", torch.tensor(mean)[None, :, None, None]
            )
            self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    @property
    def device(self) -> torch.device:
        """Return device of the model."""
        return next(self.parameters()).device

    def update_forward_args(self, **kwargs) -> None:
        """Update forward args with given keyword arguments."""
        for key, value in kwargs.items():
            self._forward_args[key] = value

    def slice_forward_args(self, index: torch.Tensor) -> None:
        """Slice forward args with given index tensor."""
        for key, value in self._forward_args.items():
            if isinstance(value, torch.Tensor):
                # Assume that all tensor kwargs are batched
                self._forward_args[key] = value[index]

    def _init_return_dict(self) -> OutputsDict:
        return {
            "class_logits": None,
            "sem_seg_logits": None,
            "sem_seg_masks": None,
            "losses": None,
        }

    def preprocess(
        self, inputs: _Inputs
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Preprocess a batch of samples.

        Args:
            inputs: Batch of samples.

        Returns:
            Preprocessed tensors: images, targets, segmentation masks (None for
            a normal classifier).
        """
        images, targets = inputs
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float()
            images /= 255
        return images, targets, None

    def postprocess(self, outputs: OutputsDict) -> OutputsDict:
        """Post-process output dictionary in-place.

        Args:
            outputs: Output dictionary.

        Returns:
            Post-processed output dictionary.
        """
        return outputs

    def forward(self, images: BatchImages, **kwargs) -> OutputsDict:
        """Forward pass.

        Args:
            images: Batched input images.

        Returns:
            Output logits.
        """
        _ = kwargs  # Unused
        if self._normalize is not None:
            images = (images - self.mean) / self.std
        class_logits = self._base_model(images, **self._forward_args)
        return_dict = self._init_return_dict()
        return_dict["class_logits"] = class_logits
        return return_dict
