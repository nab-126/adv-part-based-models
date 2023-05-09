"""ImageNet dataloader with segmentation labels."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np
import torch

from part_model.dataloader import part_imagenet
from part_model.dataloader.imagenet_class_to_metapart import (
    IMAGENET_CLASS_TO_METAPARTS,
)
from part_model.dataloader.transforms import get_imagenet_transforms
from part_model.utils import np_temp_seed
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type

logger = logging.getLogger(__name__)


class ImageNetMeta(part_imagenet.PartImageNetMeta):
    """Metadata for ImageNet dataset."""

    classes_to_num_parts = {
        "n01440764": 4,
        "n01443537": 4,
        "n01484850": 4,
        "n01491361": 4,
        "n01494475": 4,
        "n01608432": 5,
        "n01614925": 5,
        "n01630670": 4,
        "n01632458": 4,
        "n01641577": 4,
        "n01644373": 4,
        "n01644900": 4,
        "n01664065": 4,
        "n01665541": 4,
        "n01667114": 4,
        "n01667778": 4,
        "n01669191": 4,
        "n01685808": 4,
        "n01687978": 4,
        "n01688243": 4,
        "n01689811": 4,
        "n01692333": 4,
        "n01693334": 4,
        "n01694178": 4,
        "n01695060": 4,
        "n01697457": 4,
        "n01698640": 4,
        "n01728572": 2,
        "n01728920": 2,
        "n01729322": 2,
        "n01729977": 2,
        "n01734418": 2,
        "n01735189": 2,
        "n01739381": 2,
        "n01740131": 2,
        "n01742172": 2,
        "n01744401": 2,
        "n01748264": 2,
        "n01749939": 2,
        "n01753488": 2,
        "n01755581": 2,
        "n01756291": 2,
        "n01824575": 5,
        "n01828970": 5,
        "n01843065": 5,
        "n01855672": 5,
        "n02002724": 5,
        "n02006656": 5,
        "n02009229": 5,
        "n02009912": 5,
        "n02017213": 5,
        "n02025239": 5,
        "n02033041": 5,
        "n02058221": 5,
        "n02071294": 4,
        "n02085782": 4,
        "n02089867": 4,
        "n02090379": 4,
        "n02091831": 4,
        "n02092339": 4,
        "n02096177": 4,
        "n02096585": 4,
        "n02097474": 4,
        "n02098105": 4,
        "n02099601": 4,
        "n02100583": 4,
        "n02101006": 4,
        "n02101388": 4,
        "n02102040": 4,
        "n02102973": 4,
        "n02109525": 4,
        "n02109961": 4,
        "n02112137": 4,
        "n02114367": 4,
        "n02120079": 4,
        "n02124075": 4,
        "n02125311": 4,
        "n02128385": 4,
        "n02129604": 4,
        "n02130308": 4,
        "n02132136": 4,
        "n02133161": 4,
        "n02134084": 4,
        "n02134418": 4,
        "n02356798": 4,
        "n02397096": 4,
        "n02403003": 4,
        "n02408429": 4,
        "n02412080": 4,
        "n02415577": 4,
        "n02417914": 4,
        "n02422106": 4,
        "n02422699": 4,
        "n02423022": 4,
        "n02437312": 4,
        "n02441942": 4,
        "n02442845": 4,
        "n02443114": 4,
        "n02444819": 4,
        "n02447366": 4,
        "n02480495": 5,
        "n02480855": 5,
        "n02481823": 5,
        "n02483362": 5,
        "n02483708": 5,
        "n02484975": 5,
        "n02486261": 5,
        "n02486410": 5,
        "n02487347": 5,
        "n02488702": 5,
        "n02489166": 5,
        "n02490219": 5,
        "n02492035": 5,
        "n02492660": 5,
        "n02493509": 5,
        "n02493793": 5,
        "n02494079": 5,
        "n02510455": 4,
        "n02514041": 4,
        "n02536864": 4,
        "n02607072": 4,
        "n02655020": 4,
        "n02690373": 5,
        "n02701002": 3,
        "n02814533": 3,
        "n02823428": 2,
        "n02835271": 4,
        "n02930766": 3,
        "n03100240": 3,
        "n03417042": 3,
        "n03444034": 3,
        "n03445924": 3,
        "n03594945": 3,
        "n03670208": 3,
        "n03769881": 3,
        "n03770679": 3,
        "n03785016": 4,
        "n03791053": 4,
        "n03792782": 4,
        "n03937543": 2,
        "n03947888": 2,
        "n03977966": 3,
        "n03983396": 2,
        "n04037443": 3,
        "n04065272": 3,
        "n04146614": 3,
        "n04147183": 2,
        "n04252225": 3,
        "n04285008": 3,
        "n04465501": 3,
        "n04482393": 4,
        "n04483307": 2,
        "n04487081": 3,
        "n04509417": 4,
        "n04552348": 5,
        "n04557648": 2,
        "n04591713": 2,
        "n04612504": 2,
    }
    class_to_metaclass = {
        "n01440764": "Fish",
        "n01443537": "Fish",
        "n01484850": "Fish",
        "n01491361": "Fish",
        "n01494475": "Fish",
        "n01608432": "Bird",
        "n01614925": "Bird",
        "n01630670": "Reptile",
        "n01632458": "Reptile",
        "n01641577": "Reptile",
        "n01644373": "Reptile",
        "n01644900": "Reptile",
        "n01664065": "Reptile",
        "n01665541": "Reptile",
        "n01667114": "Reptile",
        "n01667778": "Reptile",
        "n01669191": "Reptile",
        "n01685808": "Reptile",
        "n01687978": "Reptile",
        "n01688243": "Reptile",
        "n01689811": "Reptile",
        "n01692333": "Reptile",
        "n01693334": "Reptile",
        "n01694178": "Reptile",
        "n01695060": "Reptile",
        "n01697457": "Reptile",
        "n01698640": "Reptile",
        "n01728572": "Snake",
        "n01728920": "Snake",
        "n01729322": "Snake",
        "n01729977": "Snake",
        "n01734418": "Snake",
        "n01735189": "Snake",
        "n01739381": "Snake",
        "n01740131": "Snake",
        "n01742172": "Snake",
        "n01744401": "Snake",
        "n01748264": "Snake",
        "n01749939": "Snake",
        "n01753488": "Snake",
        "n01755581": "Snake",
        "n01756291": "Snake",
        "n01824575": "Bird",
        "n01828970": "Bird",
        "n01843065": "Bird",
        "n01855672": "Bird",
        "n02002724": "Bird",
        "n02006656": "Bird",
        "n02009229": "Bird",
        "n02009912": "Bird",
        "n02017213": "Bird",
        "n02025239": "Bird",
        "n02033041": "Bird",
        "n02058221": "Bird",
        "n02071294": "Fish",
        "n02085782": "Quadruped",
        "n02089867": "Quadruped",
        "n02090379": "Quadruped",
        "n02091831": "Quadruped",
        "n02092339": "Quadruped",
        "n02096177": "Quadruped",
        "n02096585": "Quadruped",
        "n02097474": "Quadruped",
        "n02098105": "Quadruped",
        "n02099601": "Quadruped",
        "n02100583": "Quadruped",
        "n02101006": "Quadruped",
        "n02101388": "Quadruped",
        "n02102040": "Quadruped",
        "n02102973": "Quadruped",
        "n02109525": "Quadruped",
        "n02109961": "Quadruped",
        "n02112137": "Quadruped",
        "n02114367": "Quadruped",
        "n02120079": "Quadruped",
        "n02124075": "Quadruped",
        "n02125311": "Quadruped",
        "n02128385": "Quadruped",
        "n02129604": "Quadruped",
        "n02130308": "Quadruped",
        "n02132136": "Quadruped",
        "n02133161": "Quadruped",
        "n02134084": "Quadruped",
        "n02134418": "Quadruped",
        "n02356798": "Quadruped",
        "n02397096": "Quadruped",
        "n02403003": "Quadruped",
        "n02408429": "Quadruped",
        "n02412080": "Quadruped",
        "n02415577": "Quadruped",
        "n02417914": "Quadruped",
        "n02422106": "Quadruped",
        "n02422699": "Quadruped",
        "n02423022": "Quadruped",
        "n02437312": "Quadruped",
        "n02441942": "Quadruped",
        "n02442845": "Quadruped",
        "n02443114": "Quadruped",
        "n02444819": "Quadruped",
        "n02447366": "Quadruped",
        "n02480495": "Biped",
        "n02480855": "Biped",
        "n02481823": "Biped",
        "n02483362": "Biped",
        "n02483708": "Biped",
        "n02484975": "Biped",
        "n02486261": "Biped",
        "n02486410": "Biped",
        "n02487347": "Biped",
        "n02488702": "Biped",
        "n02489166": "Biped",
        "n02490219": "Biped",
        "n02492035": "Biped",
        "n02492660": "Biped",
        "n02493509": "Biped",
        "n02493793": "Biped",
        "n02494079": "Biped",
        "n02510455": "Quadruped",
        "n02514041": "Fish",
        "n02536864": "Fish",
        "n02607072": "Fish",
        "n02655020": "Fish",
        "n02690373": "Aeroplane",
        "n02701002": "Car",
        "n02814533": "Car",
        "n02823428": "Bottle",
        "n02835271": "Bicycle",
        "n02930766": "Car",
        "n03100240": "Car",
        "n03417042": "Car",
        "n03444034": "Car",
        "n03445924": "Car",
        "n03594945": "Car",
        "n03670208": "Car",
        "n03769881": "Car",
        "n03770679": "Car",
        "n03785016": "Bicycle",
        "n03791053": "Bicycle",
        "n03792782": "Bicycle",
        "n03937543": "Bottle",
        "n03947888": "Boat",
        "n03977966": "Car",
        "n03983396": "Bottle",
        "n04037443": "Car",
        "n04065272": "Car",
        "n04146614": "Car",
        "n04147183": "Boat",
        "n04252225": "Car",
        "n04285008": "Car",
        "n04465501": "Car",
        "n04482393": "Bicycle",
        "n04483307": "Boat",
        "n04487081": "Car",
        "n04509417": "Bicycle",
        "n04552348": "Aeroplane",
        "n04557648": "Bottle",
        "n04591713": "Bottle",
        "n04612504": "Boat",
    }

    def __init__(self) -> None:
        """See ImageNetMeta.__init__."""
        super().__init__()
        self.loader = load_imagenet
        # Plus one for the background class
        num_metaparts = (
            sum(part_imagenet.PartImageNetMeta.classes_to_num_parts.values())
            + 1
        )
        class_to_metapart = []
        for ids in IMAGENET_CLASS_TO_METAPARTS.values():
            binary_ids = [0] * num_metaparts
            for i in ids:
                binary_ids[i] = 1
            class_to_metapart.append(binary_ids)
        self.class_to_metapart_mat = torch.tensor(
            class_to_metapart, dtype=torch.float32
        )

        # Get meta-part ids from PartImageNet dataset
        metaclasses = part_imagenet.PartImageNetMeta.classes_to_num_parts
        metaclass_ids = {}
        cur_idx = 1  # Reserve 0 for background
        for metaclass, num_parts in metaclasses.items():
            metaclass_ids[metaclass] = list(range(cur_idx, cur_idx + num_parts))
            cur_idx += num_parts

        # Get a mapping from parts to meta-parts including background as the
        # 0-th index.
        num_metaparts = sum(metaclasses.values()) + 1
        bg_row = [0] * num_metaparts
        bg_row[0] = 1
        part_to_metapart = [bg_row]
        for meta_class in self.class_to_metaclass.values():
            for i in metaclass_ids[meta_class]:
                row = [0] * num_metaparts
                row[i] = 1
                part_to_metapart.append(row)
        # part_to_metapart_mat: (num_parts + 1, num_metaparts + 1)
        self.part_to_metapart_mat = torch.tensor(
            part_to_metapart, dtype=torch.float32
        )
        # part_to_metapart: array of metapart id for each part
        self.part_to_metapart = self.part_to_metapart_mat.argmax(-1)


class ImageNetSegDataset(part_imagenet.PartImageNetSegDataset):
    """PartImageNet Dataset."""

    classes_to_num_parts = ImageNetMeta.classes_to_num_parts

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
        """See PartImageNetSegDataset.__init__."""
        super(part_imagenet.PartImageNetSegDataset).__init__()
        self._root: str = root
        self._split: str = split
        self._seg_path: str = os.path.join(seg_path, split)
        self._transform = transform
        self._use_label: bool = use_label
        self._seg_type: str | None = seg_type
        self._use_atta: bool = use_atta

        self.classes: list[str] = self._list_classes()
        self.num_classes: int = len(self.classes)
        # FIXME: Replace ImageNetMeta with metaclass
        self.num_seg_labels: int = sum(
            ImageNetMeta.classes_to_num_parts[c] for c in self.classes
        )

        # Load data from specified path
        self.images, self.labels, self.masks = self._get_data()
        # Randomly shuffle data
        idx = np.arange(len(self.images))
        with np_temp_seed(seed):
            np.random.shuffle(idx)
        # Randomly drop seg masks if specified for semi-supervised training
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]


def _get_loader_sampler(args, transform, split: str):
    seg_type: str = get_seg_type(args)
    is_train: bool = split == "train"
    use_atta: bool = args.adv_train == "atta"
    imagenet_dataset = ImageNetSegDataset(
        args.data,
        args.seg_label_dir,
        split=split,
        transform=transform,
        seg_type=seg_type,
        use_label=("semi" in args.experiment) or (seg_type is None),
        seg_fraction=args.semi_label if is_train else 1.0,
        use_atta=use_atta,
    )

    sampler: torch.utils.data.Sampler | None = None
    shuffle: bool | None = is_train
    if args.distributed:
        shuffle = None
        if is_train:
            sampler = torch.utils.data.distributed.DistributedSampler(
                imagenet_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                imagenet_dataset, shuffle=False, seed=args.seed
            )

    batch_size = args.batch_size
    loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    if args.seg_labels == -1:
        pto = IMAGENET.part_to_object
        if seg_type == "part":
            seg_labels = len(pto)
        elif seg_type == "fg":
            seg_labels = 2
        elif seg_type == "group":
            seg_labels = IMAGENET.part_to_metapart_mat.shape[1]
        else:
            seg_labels = pto.max().item() + 1
        args.seg_labels = seg_labels
    args.num_classes = imagenet_dataset.num_classes
    args.input_dim = IMAGENET.input_dim
    if is_train:
        args.num_train = len(imagenet_dataset)
    return loader, sampler


def load_imagenet(args):
    """Load dataloaders for ImageNetSegDataset."""
    img_size = IMAGENET.input_dim[1]
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


IMAGENET = ImageNetMeta()
