"""Implement generic data augmentation/transforms for ImageNet."""

from __future__ import annotations

from typing import Any

import torchvision.transforms.v2 as transforms


def get_imagenet_transforms(
    is_train: bool = True,
    crop_size: int = 224,
    resize_size: int = 256,
    color_jitter: float | None = None,
    interp: str | Any = transforms.InterpolationMode.BILINEAR,
) -> transforms.Transform:
    """Get standard ImageNet data augmentation.

    Assume that input is torch.Tensor of dtype torch.uint8 and segmentation
    mask is torchvision.datapoints.Mask.

    Args:
        is_train: Use training mode. Default to True.
        crop_size: Output size after cropping. This is final output size.
            Default to 224.
        resize_size: Size to resize to before cropping. Default to 256.
        color_jitter: Amount of color jitter to apply. If None, no color jitter.
            Default to None.
        interp: Interpolation mode for resizing. Default to bilinear.

    Returns:
        torchvision Transform object.
    """
    if is_train:
        # Build augmentation
        augs_list = [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.08, 1),
                ratio=(0.75, 1.3333),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
        ]
        if color_jitter is not None:
            augs_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                )
            )
        augs = transforms.Compose(augs_list)
        return augs

    augs = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=interp, antialias=True
            ),
            transforms.CenterCrop(crop_size),
        ]
    )
    return augs


def get_paco_transforms(
    is_train: bool = True,
    crop_size: int = 224,
    resize_size: int = 256,
    color_jitter: float | None = None,
    interp: str | Any = transforms.InterpolationMode.BILINEAR,
) -> transforms.Transform:
    """Get data augmentation for the PACO dataset.

    Assume that input is torch.Tensor of dtype torch.uint8 and segmentation
    mask is torchvision.datapoints.Mask.

    Args:
        is_train: Use training mode. Default to True.
        crop_size: Output size after cropping. This is final output size.
            Default to 224.
        resize_size: Size to resize to before cropping. Default to 256.
        color_jitter: Amount of color jitter to apply. If None, no color jitter.
            Default to None.
        interp: Interpolation mode for resizing. Default to bilinear.

    Returns:
        torchvision Transform object.
    """
    if is_train:
        # Build augmentation
        augs_list = [
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.5, 1),
                ratio=(0.75, 1.3333),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(0.5),
        ]
        if color_jitter is not None:
            augs_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                )
            )
        augs = transforms.Compose(augs_list)
        return augs

    augs = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=interp, antialias=True
            ),
            transforms.CenterCrop(crop_size),
        ]
    )
    return augs


DATASET_TO_TRANSFORMS = {
    "paco": get_paco_transforms,
    "imagenet": get_imagenet_transforms,
    "part-imagenet": get_imagenet_transforms,
}
