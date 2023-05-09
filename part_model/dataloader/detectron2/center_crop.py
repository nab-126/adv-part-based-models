"""Implement center crop."""

import numpy as np
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import CropTransform, Transform, TransformList


class CenterCrop(Augmentation):
    """Center crop for evaluation."""

    def __init__(self, crop_size: tuple[int]) -> None:
        """Initialize the center crop transform.

        Args:
            crop_size: target image (height, width).
        """
        super().__init__()
        self._crop_size = crop_size

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]

        # Add random crop if the image is scaled up.
        offset = np.subtract(input_size, self._crop_size)
        offset = np.maximum(offset, 0) / 2
        offset = np.round(offset).astype(int)
        return CropTransform(
            offset[1],
            offset[0],
            self._crop_size[1],
            self._crop_size[0],
            input_size[1],
            input_size[0],
        )

    # pylint: disable=arguments-differ
    def get_transform(self, image: np.ndarray) -> TransformList:
        """Get the transform."""
        transforms = [self._get_crop(image)]
        return TransformList(transforms)
