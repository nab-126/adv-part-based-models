"""PACO dataset."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, List, Tuple, Union

import torch
import torchvision
from PIL import Image

from DINO.datasets.coco import ConvertCocoPolysToMask
from part_model.dataloader.segmentation_transforms import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from part_model.utils.colors import COLORMAP
from part_model.utils.eval_sampler import DistributedEvalSampler
from part_model.utils.image import get_seg_type

import torch.nn.functional as F

class PacoMeta(torchvision.datasets.VisionDataset):
    """Metadata for PACO dataset."""

    OBJ_CLASSES = [
        "trash_can","handbag","ball","basket","bicycle","book","bottle","bowl","can","car_(automobile)","carton","cellular_telephone","chair","cup","dog","drill","drum_(musical_instrument)","glass_(drink_container)","guitar","hat","helmet","jar","knife","laptop_computer","mug","pan_(for_cooking)","plate","remote_control","scissors","shoe","slipper_(footwear)","stool","table","towel","wallet","watch","wrench","belt","bench","blender","box","broom","bucket","calculator","clock","crate","earphone","fan","hammer","kettle","ladder","lamp","microwave_oven","mirror","mouse_(computer_equipment)","napkin","newspaper","pen","pencil","pillow","pipe","pliers","plastic_bag","scarf","screwdriver","soap","sponge","spoon","sweater","tape_(sticky_cloth_or_paper)","telephone","television_set","tissue_paper","tray","vase",
    ]
    ATTRIBUTES = ['black', 'light_blue', 'blue', 'dark_blue', 'light_brown', 'brown', 'dark_brown', 'light_green', 'green', 'dark_green', 'light_grey', 'grey', 'dark_grey', 'light_orange', 'orange', 'dark_orange', 'light_pink', 'pink', 'dark_pink', 'light_purple', 'purple', 'dark_purple', 'light_red', 'red', 'dark_red', 'white', 'light_yellow', 'yellow', 'dark_yellow', 'other(color)', 'plain', 'striped', 'dotted', 'checkered', 'woven', 'studded', 'perforated', 'floral', 'other(pattern_marking)', 'logo', 'text', 'stone', 'wood', 'rattan', 'fabric', 'crochet', 'wool', 'leather', 'velvet', 'metal', 'paper', 'plastic', 'glass', 'ceramic', 'other(material)', 'opaque', 'translucent', 'transparent', 'other(transparency)']
    COLOR_ATTRIBUTES = ATTRIBUTES[:30] 
    PATTERN_ATTRIBUTES = ATTRIBUTES[30:41] 
    MATERIAL_ATTRIBUTES = ATTRIBUTES[41:55] 
    REFLECTANCE_ATTRIBUTES = ATTRIBUTES[55:59] 

    normalize = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    input_dim = (3, 224, 224)
    colormap = COLORMAP

    def __init__(
        self,
        root: str=None,
        ann_path: str=None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self.num_classes: int = len(self.OBJ_CLASSES)
        self.loader = load_paco

        part_to_object = [0, 9, 12, 32, 12, 38, 12, 18, 27, 23, 38, 70, 11, 29, 37, 57, 6, 7, 44, 16, 42, 1, 47, 26, 71, 3, 8, 24, 21, 65, 13, 49, 73, 12, 25, 39, 51, 17, 23, 4, 70, 11, 22, 47, 39, 61, 28, 62, 6, 18, 7, 15, 58, 16, 68, 0, 63, 42, 1, 26, 43, 8, 54, 24, 21, 65, 33, 49, 17, 74, 14, 33, 8, 42, 1, 24, 6, 45, 7, 21, 25, 40, 26, 65, 73, 17, 71, 0, 10, 3, 67, 47, 18, 41, 41, 37, 35, 51, 9, 70, 27, 71, 11, 44, 39, 51, 23, 46, 49, 23, 47, 57, 65, 10, 6, 65, 6, 35, 44, 57, 65, 6, 60, 52, 42, 21, 39, 3, 5, 68, 39, 44, 35, 52, 52, 4, 32, 24, 14, 46, 59, 37, 58, 14, 29, 48, 20, 47, 9, 58, 28, 18, 51, 44, 34, 39, 14, 74, 50, 31, 4, 37, 53, 63, 4, 9, 48, 57, 35, 44, 62, 6, 15, 48, 28, 41, 64, 42, 1, 36, 61, 3, 24, 45, 21, 9, 65, 13, 49, 22, 67, 25, 39, 74, 4, 14, 48, 36, 16, 4, 46, 9, 18, 29, 6, 62, 68, 33, 37, 18, 0, 9, 46, 8, 42, 62, 1, 24, 6, 34, 7, 21, 39, 16, 17, 0, 13, 32, 49, 19, 45, 20, 40, 52, 25, 3, 10, 73, 26, 29, 30, 61, 61, 43, 18, 23, 65, 6, 0, 29, 58, 54, 32, 38, 12, 14, 31, 8, 45, 40, 21, 25, 10, 0, 49, 47, 29, 30, 19, 54, 20, 47, 9, 27, 23, 37, 42, 16, 41, 35, 16, 9, 47, 74, 6, 67, 74, 65, 14, 68, 14, 60, 60, 73, 29, 30, 5, 4, 0, 47, 44, 18, 51, 19, 37, 8, 65, 6, 65, 29, 12, 50, 54, 8, 42, 19, 1, 24, 7, 20, 21, 25, 9, 26, 17, 73, 16, 0, 13, 32, 3, 65, 41, 6, 47, 72, 69, 9, 66, 9, 4, 70, 11, 23, 28, 54, 39, 12, 38, 31, 9, 4, 4, 51, 51, 51, 41, 64, 32, 68, 65, 6, 45, 18, 40, 52, 25, 71, 10, 3, 54, 9, 65, 6, 12, 68, 46, 12, 9, 39, 65, 49, 6, 9, 4, 31, 50, 21, 12, 19, 35, 20, 30, 37, 12, 32, 38, 18, 47, 39, 51, 49, 12, 38, 14, 9, 9, 10, 14, 33, 8, 62, 24, 56, 21, 10, 29, 52, 67, 57, 64, 29, 30, 29, 6, 52, 65, 71, 32, 10, 50, 4, 23, 9, 9, 52, 29, 30, 39, 19, 20, 29, 12, 9, 0, 32, 4, 35, 9, 9, 9, 9, 54, 68, 1]
        part_to_class = [[0] * (self.num_classes + 1) for _ in range(len(part_to_object))]
        for part_index, part in enumerate(part_to_object):
            part_to_class[part_index][part] = 1
        self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)
        self.part_to_class = torch.tensor(part_to_class, dtype=torch.float32)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def _load_target_attributes(self, id: int) -> List[Any]:
        image_ann = self.coco.loadImgs(id)[0]
        color_attribute = image_ann["color_attribute"] if "color_attribute" in image_ann else None
        pattern_attribute = image_ann["pattern_attribute"] if "pattern_attribute" in image_ann else None
        material_attribute = image_ann["material_attribute"] if "material_attribute" in image_ann else None
        reflectance_attribute = image_ann["reflectance_attribute"] if "reflectance_attribute" in image_ann else None
        return color_attribute, pattern_attribute, material_attribute, reflectance_attribute

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target_attributes = self._load_target_attributes(id)
        
        self.coco.loadImgs(id)[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, target_attributes

    def __len__(self) -> int:
        return len(self.ids)

class PacoDataset(PacoMeta):
    """PACO Dataset."""

    def __init__(
        self,
        root: str = "~/data/",
        seg_path: str = "~/data/",
        ann_path: str = "~/data/",
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
        # super(PacoDataset, self).__init__(root, ann_path)
        super().__init__(root, ann_path)

        self._root: str = root
        self._split: str = split
        self._seg_path: str = seg_path
        self._transform = transform
        self._use_label: bool = use_label
        self._seg_type: str | None = seg_type
        self._use_atta: bool = use_atta

        self.prepare = ConvertCocoPolysToMask(False)
        self.num_classes: int = len(self.OBJ_CLASSES)

        # Load data from specified path
        # self.images, self.labels, self.masks = self._get_data()
        # TODO: Randomly shuffle data (currently deprecated)
        # idx = np.arange(len(self.images))
        # with np_temp_seed(seed):
        #     np.random.shuffle(idx)
        # # Randomly drop seg masks if specified for semi-supervised training
        # self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]

        # Create matrix that maps part segmentation to object segmentation
        # part_to_object = [0]
        # self.part_to_class = [[0] * (self.num_classes + 1)]
        # self.part_to_class[0][0] = 1
        # for i, label in enumerate(self.classes):
        #     part_to_object.extend([i + 1] * self.CLASSES[label])
        #     base = [0] * (self.num_classes + 1)
        #     base[i + 1] = 1
        #     self.part_to_class.extend([base] * self.CLASSES[label])
        # self.part_to_object = torch.tensor(part_to_object, dtype=torch.long)

        # bounding boxes
        with open(ann_path) as f:
            annotations = json.load(f)

        self.num_seg_labels: int = len(
            annotations["part_categories"]
        )  # for background class

        self.imageid_to_label = {}
        self.imageid_to_seg_filename = {}
        for ann in annotations["images"]:
            image_id = ann["id"]
            supercategory = ann["supercategory"]
            class_label = self.OBJ_CLASSES.index(supercategory)
            self.imageid_to_label[image_id] = class_label
            seg_filename = ann["seg_filename"]
            self.imageid_to_seg_filename[image_id] = seg_filename

    def one_hot_encode_attributes(self, attributes: list[int], num_attribute_classes) -> torch.Tensor:
        """Transform attributes to tensor.

        Args:
            attributes: List of attributes to transform.

        Returns:
            Tensor of attributes.
        """
        # attributes should be a long tensor with zeros
        attributes = torch.zeros(num_attribute_classes, dtype=torch.long)
        if attributes is not None:
            return attributes
        for attribute_id in attributes:
            attributes[attribute_id] = 1
        return attributes
        # if attributes is None:
        #     attributes = torch.zeros(num_attribute_classes)
        # else:
        #     attributes = torch.tensor(attributes, dtype=torch.long)
        #     attributes = F.one_hot(attributes, num_classes=num_attribute_classes)
        # return attributes
    
    def transform_attributes(self, attributes: list[int]) -> torch.Tensor:
        """Transform attributes to tensor.

        Args:
            attributes: List of attributes to transform.

        Returns:
            Tensor of attributes.
        """
        color_attribute, pattern_attribute, material_attribute, reflectance_attribute = attributes
        color_attribute = self.one_hot_encode_attributes(color_attribute, len(self.COLOR_ATTRIBUTES))
        pattern_attribute = self.one_hot_encode_attributes(pattern_attribute, len(self.PATTERN_ATTRIBUTES))
        material_attribute = self.one_hot_encode_attributes(material_attribute, len(self.MATERIAL_ATTRIBUTES))
        reflectance_attribute = self.one_hot_encode_attributes(reflectance_attribute, len(self.REFLECTANCE_ATTRIBUTES))
        return color_attribute, pattern_attribute, material_attribute, reflectance_attribute

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

        try:
            img, bbox_target, attribute_target = super(PacoDataset, self).__getitem__(index)
        except:
            print("Error index: {}".format(index))
            index += 1
            img, bbox_target, attribute_target = super(PacoDataset, self).__getitem__(index)

        image_id = self.ids[index]
        _label = self.imageid_to_label[image_id]
        _label = torch.tensor(_label, dtype=torch.long)
        seg_filename = self.imageid_to_seg_filename[image_id]
        seg_mask_path = os.path.join(self._seg_path, seg_filename)
        seg_mask_target = Image.open(seg_mask_path)

        bbox_target = {"image_id": image_id, "annotations": bbox_target}
        img, bbox_target = self.prepare(img, bbox_target)
        attribute_target = self.transform_attributes(attribute_target)
        width, height = img.size

        if self._transform is not None:
            tf_out = self._transform(img, seg_mask_target)

            if len(tf_out) == 3:
                # In ATTA, transform params are also returned
                img, seg_mask_target, params = tf_out
                atta_data = [torch.tensor(index), torch.tensor((height, width))]
                for p in params:
                    atta_data.append(torch.tensor(p))
            else:
                img, seg_mask_target = tf_out

        # add image to return items
        return_items.append(img)

        # add class label to return items
        if self._use_label:
            return_items.append(_label)

        # add segmentation mask to return items
        # return_items.append(attribute_target)
        # print('attribute_target', attribute_target)
        # for atta in attribute_target:
            # print('atta', atta.shape)
            # return_items.append(atta)
        # print()
        return_items.extend(attribute_target)
        # if self._seg_type is not None:
        #     pass
        #     # TODO: currently deprecated for PACO
        #     # if self._seg_type == "object":
        #     #     seg_mask_target = self.part_to_object[seg_mask_target]
        #     # elif self._seg_type == "fg":
        #     #     seg_mask_target = (seg_mask_target > 0).long()
        #     # if index in self.seg_drop_idx:
        #     #     # Drop segmentation mask by setting all pixels to -1 to ignore
        #     #     # later at loss computation
        #     #     seg_mask_target.mul_(0).add_(-1)
        #     return_items.append(seg_mask_target)
        # # else:
        # #     seg_mask_target = None
        

        # PACO bbox currently deprecated
        # add bbox target to return items
        # normalize bbox target
        # h, w = img.shape[-2:]
        # if "boxes" in bbox_target:
        #     boxes = bbox_target["boxes"]
        #     boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        #     boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        #     bbox_target["boxes"] = boxes
        # return_items.append(bbox_target)

        # add atta data to return items
        if atta_data is not None:
            for ad in atta_data:
                return_items.append(ad)

        return return_items


def get_loader_sampler(args, transform, split: str):
    seg_type: str = get_seg_type(args)
    is_train: bool = split == "train"
    use_atta: bool = args.adv_train == "atta"

    img_folder = os.path.join(
        args.data, "PartSegmentations", "All", split, "images"
    )
    seg_folder = os.path.join(
        args.data, "PartSegmentations", "All", split, "seg_masks"
    )
    ann_file_path = os.path.join(
        args.data, "PartSegmentations", "All", f"{split}.json"
    )

    paco_dataset = PacoDataset(
        img_folder,
        seg_folder,
        ann_file_path,
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
                paco_dataset,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
        else:
            # Use distributed sampler for validation but not testing
            sampler = DistributedEvalSampler(
                paco_dataset, shuffle=False, seed=args.seed
            )

    batch_size = args.batch_size

    loader = torch.utils.data.DataLoader(
        paco_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )

    # if args.seg_labels == -1:
    #     pto = PART_IMAGENET.part_to_object
    #     if seg_type == "part":
    #         seg_labels = len(pto)
    #     elif seg_type == "fg":
    #         seg_labels = 2
    #     else:
    #         seg_labels = pto.max().item() + 1
    #     args.seg_labels = seg_labels
    args.seg_labels = paco_dataset.num_seg_labels

    args.num_classes = paco_dataset.num_classes
    args.input_dim = PacoMeta.input_dim
    if is_train:
        args.num_train = len(paco_dataset)

    return loader, sampler


def load_paco(args):
    img_size = PACO.input_dim[1]
    use_atta: bool = args.adv_train == "atta"

    train_transforms = Compose(
        [
            RandomResizedCrop(
                img_size, return_params=use_atta, scale=(0.5, 1.0)
            ),
            RandomHorizontalFlip(0.5, return_params=use_atta),
            ToTensor(),
        ]
    )

    val_transforms = Compose(
        [
            Resize(int(img_size)),
            CenterCrop(img_size),
            ToTensor(),
        ]
    )

    train_loader, train_sampler = get_loader_sampler(
        args, train_transforms, "train"
    )
    val_loader, _ = get_loader_sampler(args, val_transforms, "val")
    test_loader, _ = get_loader_sampler(args, val_transforms, "test")

    return train_loader, train_sampler, val_loader, test_loader


PACO = PacoMeta()
