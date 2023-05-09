"""Custom dataset mapper for semantic segmentation task.

This code is based on MaskFormerSemanticDatasetMapper.
"""

import copy
import logging

import torch
import torchvision
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import BitMasks, Boxes, Instances
from torch.nn import functional as F

from part_model.dataloader.transforms import DATASET_TO_TRANSFORMS

__all__ = ["CustomSemanticDatasetMapper"]


class CustomSemanticDatasetMapper:
    """Default part segmentation dataset mapper for part-based models.

    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        seg_include_bg: bool = False,
        bg_label: int = -1,
        color_jitter: float | None = 0.3,
        seg_label_map: None | torch.Tensor = None,
        meta_ignore_label: int | None = None,
        meta_bg_label: int | None = None,
    ):
        """Initialize a CustomSemanticDatasetMapper.

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            seg_include_bg: If True, include background class as a separate
                class. Otherwise, background class is set as ignore_label.
            bg_label: The class id of background class.
            color_jitter: If float is given, apply color jittering to the image.
            seg_label_map: A mapping for grouping some of segmentation labels
                together, e.g., grouping class-specific parts into meta-parts.
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self._seg_include_bg = seg_include_bg
        self._bg_label = bg_label
        self._color_jitter = color_jitter
        self._seg_label_map = seg_label_map
        self._meta_ignore_label = meta_ignore_label or ignore_label
        self._meta_bg_label = meta_bg_label or bg_label

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info("Augmentations used in %s: %s", mode, augmentations)

    @classmethod
    def from_config(
        cls,
        cfg,
        is_train=True,
        seg_include_bg: bool = False,
        color_jitter: float | None = 0.3,
    ):
        """Initialize a CustomSemanticDatasetMapper from config.

        Args:
            cfg: detectron2 config.
            is_train: Set to True if use mapper for training. Defaults to True.

        Returns:
            Dictionary args for init.
        """
        dataset_names = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
        meta = MetadataCatalog.get(dataset_names[0])
        resize_size = (
            cfg.INPUT.MIN_SIZE_TRAIN[0] if is_train else cfg.INPUT.MIN_SIZE_TEST
        )
        assert isinstance(resize_size, int), "resize_size must be int!"
        crop_size = cfg.INPUT.CROP.SIZE[0]
        assert isinstance(crop_size, int), "crop_size must be int!"
        augs = DATASET_TO_TRANSFORMS[cfg.DATASETS.dataset](
            crop_size=crop_size,
            resize_size=resize_size,
            is_train=is_train,
            color_jitter=color_jitter,
        )
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": meta.ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "seg_include_bg": seg_include_bg,
            "bg_label": meta.bg_label,
            "meta_ignore_label": meta.model_ignore_label,
            "meta_bg_label": meta.model_bg_label,
        }
        return ret

    def _shift_seg_idx(self, sem_seg_gt, is_meta=False):
        bg_label = self._meta_bg_label if is_meta else self._bg_label
        ignore_label = self._meta_ignore_label if is_meta else self.ignore_label
        sem_seg_gt -= 1
        if self._seg_include_bg:
            sem_seg_gt[sem_seg_gt == -1] = bg_label
        else:
            sem_seg_gt[sem_seg_gt == -1] = ignore_label
        sem_seg_gt[sem_seg_gt == -2] = ignore_label
        return sem_seg_gt

    def __call__(self, dataset_dict):
        """Map a dataset dict into a format used by the model.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format
        )
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt, sem_seg_gt_fg = None, None
        if "sem_seg_file_name" in dataset_dict:
            # FIXME: ? PyTorch transformation not implemented for uint16, so converting
            # it to double first
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name")
            ).astype("double")

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation "
                f"dataset {dataset_dict['file_name']}."
            )

        image = torch.tensor(image).permute(2, 0, 1)
        sem_seg_gt = torchvision.datapoints.Mask(sem_seg_gt)
        image, sem_seg_gt = self.tfm_gens(image, sem_seg_gt)
        assert image.shape[-2:] == sem_seg_gt.shape[-2:]

        sem_seg_gt = sem_seg_gt.long()
        if self._seg_label_map is not None:
            sem_seg_gt_fg = sem_seg_gt.clone()
            sem_seg_gt_fg = self._shift_seg_idx(sem_seg_gt_fg, is_meta=False)
            # Re-group segmentation labels
            sem_seg_gt = self._seg_label_map[sem_seg_gt].long()
            assert sem_seg_gt.shape == sem_seg_gt_fg.shape
        sem_seg_gt = self._shift_seg_idx(sem_seg_gt, is_meta=True)

        # Pad image and segmentation label here!
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()
            if sem_seg_gt_fg is not None:
                sem_seg_gt_fg = F.pad(
                    sem_seg_gt_fg, padding_size, value=self.ignore_label
                ).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Torch dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of
        # pickle & mp.Queue. Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            instances = Instances(image_shape)
            classes = sem_seg_gt.unique()
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = classes

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            # Add object class
            # NOTE: There is only one object per image but we need to add an
            # attribute with length equal to the number of "instance" to make
            # Instance object happy.
            instances.obj_class = torch.tensor(
                [dataset_dict["obj_class"]] * len(classes), dtype=torch.int64
            )
            dataset_dict["instances"] = instances
            dataset_dict["sem_seg"] = (
                sem_seg_gt if sem_seg_gt_fg is None else sem_seg_gt_fg
            )

        return dataset_dict
