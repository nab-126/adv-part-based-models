"""PartImageNet dataset."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision
from detectron2.data import detection_utils as utils
from torch.utils import data

from part_model.dataloader.transforms import get_imagenet_transforms
from part_model.utils import np_temp_seed
from part_model.utils.colors import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type

logger = logging.getLogger(__name__)


class PartImageNetMeta(data.Dataset):
    """Metadata for PartImageNet dataset."""

    classes_to_num_parts = {
        "Aeroplane": 5,
        "Bicycle": 4,
        "Biped": 5,
        "Bird": 5,
        "Boat": 2,
        "Bottle": 2,
        "Car": 3,
        "Fish": 4,
        "Quadruped": 4,
        "Reptile": 4,
        "Snake": 2,
    }
    normalize = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    input_dim = (3, 224, 224)
    colormap = COLORMAP

    def __init__(self) -> None:
        """Initialize PartImageNet metadata class."""
        super().__init__()
        num_classes = len(self.classes_to_num_parts)
        part_to_class = [[0] * (num_classes + 1)]
        part_to_class[0][0] = 1
        part_to_object = [0]
        for i, label in enumerate(self.classes_to_num_parts):
            part_to_object.extend([i + 1] * self.classes_to_num_parts[label])
            base = [0] * (num_classes + 1)
            base[i + 1] = 1
            part_to_class.extend([base] * self.classes_to_num_parts[label])
        self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)
        self.part_to_class = torch.tensor(part_to_class, dtype=torch.float32)
        self.num_seg_labels = sum(self.classes_to_num_parts.values())
        self.num_classes = len(self.classes_to_num_parts)
        self.loader = load_part_imagenet

    def __getitem__(self, index: int):
        """Get sample at index."""
        _ = index  # Unused
        raise NotImplementedError("__getitem__ must be implemented!")


class PartImageNetSegDataset(PartImageNetMeta):
    """PartImageNet Dataset."""

    def __init__(
        self,
        root: str = "~/data/",
        seg_path: str = "~/data/",
        split: str = "train",
        transform: Callable[..., Any] = None,
        use_label: bool = False,
        seg_type: str | None = None,
        seg_fraction: float = 1.0,
        seed: int = 0,
        use_atta: bool = False,
    ) -> None:
        """Load our processed Part-ImageNet dataset.

        Args:
            root: Path to root directory.
            seg_path: Path to segmentation labels.
            split: Data split to load. Defaults to "train".
            transform: Transformations to apply to the images (and
                the segmentation masks if applicable). Defaults to None.
            use_label: Whether to yield class label. Defaults to False.
            seg_type: Specify types of segmentation to load
                ("part", "object", "fg", or None). Defaults to "part".
            seg_fraction: Fraction of segmentation mask to
                provide. The dropped masks are set to all -1. Defaults to 1.
            seed: Random seed. Defaults to 0.
            use_atta: If True, use ATTA (fast adversarial training) and return
                transform params during training.
        """
        super().__init__()
        self._root: str = root
        self._split: str = split
        self._seg_path: str = os.path.join(seg_path, split)
        self._transform = transform
        self._use_label: bool = use_label
        self._seg_type: str | None = seg_type
        self._use_atta: bool = use_atta
        self.part_to_metapart = None

        self.classes: list[str] = self._list_classes()
        self.num_classes: int = len(self.classes)
        self.num_seg_labels: int = sum(
            self.classes_to_num_parts[c] for c in self.classes
        )

        # Load data from specified path
        self.images, self.labels, self.masks = self._get_data()
        # Randomly shuffle data
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
        # Randomly drop seg masks if specified for semi-supervised training
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Get sample at index.

        Args:
            index: Index of data sample to retrieve.

        Returns:
            Image, segmentation label (optional), class label (optional),
            transform params (optional).
        """
        # Collect variables to return
        return_items: list[Any] = []
        atta_data: list[torch.Tensor] | None = None
        _img = utils.read_image(self.images[index], format="RGB")
        _target = utils.read_image(self.masks[index]).astype("long")
        _, height, width = _img.shape

        if self._transform is not None:
            _target = torchvision.datapoints.Mask(_target)
            _img = torch.tensor(_img).permute(2, 0, 1)
            assert _img.shape[-2:] == _target.shape[-2:]
            tf_out = self._transform(_img, _target)
            if len(tf_out) == 3:
                # In ATTA, transform params are also returned
                _img, _target, params = tf_out
                atta_data = [torch.tensor(index), torch.tensor((height, width))]
                for p in params:
                    atta_data.append(torch.tensor(p))
            else:
                _img, _target = tf_out
        return_items.append(_img)

        # Add class label if applicable
        if self._use_label:
            _label = self.labels[index]
            return_items.append(_label)

        # Add segmentation mask if applicable
        if self._seg_type is not None:
            if self._seg_type == "object":
                _target = self.part_to_object[_target]
            elif self._seg_type == "fg":
                _target = (_target > 0).long()
            elif self._seg_type == "group":
                # Group class-specific parts into meta-parts
                _target = self.part_to_metapart[_target]
            if index in self.seg_drop_idx:
                # Drop segmentation mask by setting all pixels to -1 to ignore
                # later at loss computation
                _target.mul_(0).add_(-1)
            return_items.append(_target)

        if atta_data is not None:
            for ad in atta_data:
                return_items.append(ad)
        return return_items

    def _get_data(self):
        images, labels, masks = [], [], []
        for label_idx, label in enumerate(self.classes):
            img_path = os.path.join(self._root, "JPEGImages")
            part_path = os.path.join(self._seg_path, label)
            # Read file names
            with open(
                f"{self._seg_path}/{label}.txt", "r", encoding="utf-8"
            ) as fns:
                filenames = sorted([f.strip() for f in fns.readlines()])
            images.extend([f"{img_path}/{f}.JPEG" for f in filenames])

            # Find groundtruth masks in either .png or .tif format
            for filename in filenames:
                gt_file = f"{part_path}/{filename.split('/')[1]}.png"
                if not os.path.exists(gt_file):
                    gt_file = f"{part_path}/{filename.split('/')[1]}.tif"
                assert os.path.exists(
                    gt_file
                ), f"Ground truth file {gt_file} does not exist!"
                masks.append(gt_file)

            labels.extend([label_idx] * len(filenames))
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, masks

    def _list_classes(self):
        dirs = os.listdir(self._seg_path)
        dirs = [
            d for d in dirs if os.path.isdir(os.path.join(self._seg_path, d))
        ]
        return sorted(dirs)

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.images)


def _get_loader_sampler(args, transform, split: str):
    """Get data loader and sampler for training or validation.

    Args:
        args: Command line arguments.
        transform: Transform to apply to data.
        split: "train" or "val".

    Returns:
        Data loader and sampler.
    """
    seg_type: str = get_seg_type(args)
    is_train: bool = split == "train"
    use_atta: bool = args.adv_train == "atta"

    part_imagenet_dataset = PartImageNetSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
        use_atta=use_atta,
    )

    sampler: Optional[torch.utils.data.Sampler] = None
    shuffle: Optional[bool] = is_train
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                part_imagenet_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                part_imagenet_dataset, shuffle=False, seed=args.seed
            )

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    if args.seg_labels == -1:
        pto = PART_IMAGENET.part_to_object
        if seg_type == "part":
            seg_labels = len(pto)
        elif seg_type == "fg":
            seg_labels = 2
        else:
            seg_labels = pto.max().item() + 1
        args.seg_labels = seg_labels
    args.num_classes = part_imagenet_dataset.num_classes
    args.input_dim = PartImageNetMeta.input_dim
    if is_train:
        args.num_train = len(part_imagenet_dataset)
    return loader, sampler


def load_part_imagenet(args):
    """Load Part-ImageNet dataset."""
    img_size = PART_IMAGENET.input_dim[1]
    train_transforms = get_imagenet_transforms(
        is_train=True,
        crop_size=img_size,
        resize_size=int(img_size * 256 / 224),
        color_jitter=args.color_jitter,
    )
    logger.info("Train transforms: %s", train_transforms)
    val_transforms = get_imagenet_transforms(
        is_train=False,
        crop_size=img_size,
        resize_size=int(img_size * 256 / 224),
        color_jitter=args.color_jitter,
    )
    train_loader, train_sampler = _get_loader_sampler(
        args, train_transforms, "train"
    )
    val_loader, _ = _get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = _get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PART_IMAGENET = PartImageNetMeta()
