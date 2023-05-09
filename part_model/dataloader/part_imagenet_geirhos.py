import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from part_model.utils import get_seg_type, np_temp_seed
from part_model.utils.colors import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type

from .segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

CLASSES = {
    "Quadruped": 4,
    "Biped": 5,
    "Fish": 4,
    "Bird": 5,
    "Snake": 2,
    "Reptile": 4,
    "Car": 3,
    "Bicycle": 4,
    "Boat": 2,
    "Aeroplane": 5,
    "Bottle": 2,
}


class PartImageNetGeirhosSegDataset(data.Dataset):
    def __init__(
        self,
        root,
        seg_path,
        split="train",
        transform=None,
        use_label=False,
        seg_type=None,
        seg_fraction=1.0,
        seed=0,
    ):
        """Load our processed Part-ImageNet-Geirhos dataset

        Args:
            root (str): Path to root directory
            split (str, optional): Data split to load. Defaults to 'train'.
            transform (optional): Transformations to apply to the images (and
                the segmentation masks if applicable). Defaults to None.
            use_label (bool, optional): Whether to yield class label. Defaults to False.
            seg_type (str, optional): Specify types of segmentation to load
                ('part', 'object', or None). Defaults to 'part'.
            seg_fraction (float, optional): Fraction of segmentation mask to
                provide. The dropped masks are set to all -1. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 0.
        """
        self.root = root
        self.split = split
        self.path = os.path.join(seg_path, split)
        self.transform = transform
        self.use_label = use_label
        self.seg_type = seg_type

        self.classes = self._list_classes()
        self.num_classes = len(self.classes)
        self.num_seg_labels = sum([CLASSES[c] for c in self.classes])

        self.images, self.labels, self.masks = self._get_data()
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]

        # Create matrix that maps part segmentation to object segmentation
        part_to_object = [0]
        self.part_to_class = [[0] * (self.num_classes + 1)]
        self.part_to_class[0][0] = 1
        for i, label in enumerate(self.classes):
            part_to_object.extend([i + 1] * CLASSES[label])
            base = [0] * (self.num_classes + 1)
            base[i + 1] = 1
            self.part_to_class.extend([base] * CLASSES[label])
        self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.masks[index])

        if self.transform is not None:
            _img, _target = self.transform(_img, _target)

        if self.seg_type is not None:
            if self.seg_type == "object":
                _target = self.part_to_object[_target]
            elif self.seg_type == "fg":
                _target = (_target > 0).long()
            if index in self.seg_drop_idx:
                # Drop segmentation mask by setting all pixels to -1 to ignore
                # later at loss computation
                _target.mul_(0).add_(-1)

            if self.use_label:
                _label = self.labels[index]
                return _img, _label, _target
            return _img, _target

        if self.use_label:
            _label = self.labels[index]
            return _img, _label
        return _img

    def _get_data(self):
        images, labels, masks = [], [], []
        for l, label in enumerate(self.classes):
            img_path = os.path.join(self.root, "StyleJPEGImages")
            part_path = os.path.join(self.path, label)
            # Read file names
            with open(f"{self.path}/{label}.txt", "r") as fns:
                filenames = sorted([f.strip() for f in fns.readlines()])
            images.extend([f"{img_path}/{f}.png" for f in filenames])
            masks.extend([f'{part_path}/{f.split("/")[1]}.tif' for f in filenames])
            labels.extend([l] * len(filenames))
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, masks

    def _list_classes(self):
        dirs = os.listdir(self.path)
        dirs = [d for d in dirs if os.path.isdir(os.path.join(self.path, d))]
        return sorted(dirs)

    def __len__(self):
        return len(self.images)


def get_loader_sampler(args, transform, split, distributed_sampler=True):
    seg_type = get_seg_type(args)
    is_train = split == "train"

    part_imagenet_geirhos_dataset = PartImageNetGeirhosSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
    )

    sampler = None
    if args.distributed and distributed_sampler:
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                part_imagenet_geirhos_dataset
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(part_imagenet_geirhos_dataset)

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        part_imagenet_geirhos_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    # TODO: can we make this cleaner?
    PART_IMAGENET_GEIRHOS["part_to_class"] = part_imagenet_geirhos_dataset.part_to_class
    PART_IMAGENET_GEIRHOS["num_classes"] = part_imagenet_geirhos_dataset.num_classes
    PART_IMAGENET_GEIRHOS[
        "num_seg_labels"
    ] = part_imagenet_geirhos_dataset.num_seg_labels

    setattr(args, "num_classes", part_imagenet_geirhos_dataset.num_classes)
    pto = part_imagenet_geirhos_dataset.part_to_object
    if seg_type == "part":
        seg_labels = len(pto)
    elif seg_type == "fg":
        seg_labels = 2
    else:
        seg_labels = pto.max().item() + 1
    setattr(args, "seg_labels", seg_labels)

    return loader, sampler


def load_part_imagenet_geirhos(args):

    img_size = PART_IMAGENET_GEIRHOS["input_dim"][1]

    train_transforms = Compose(
        [
            RandomResizedCrop(img_size),
            RandomHorizontalFlip(0.5),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            Resize(int(img_size * 256 / 224)),
            CenterCrop(img_size),
            ToTensor(),
        ]
    )

    train_loader, train_sampler = get_loader_sampler(args, train_transforms, "train")
    val_loader, _ = get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PART_IMAGENET_GEIRHOS = {
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "loader": load_part_imagenet_geirhos,
    "input_dim": (3, 224, 224),
    "colormap": COLORMAP,
}