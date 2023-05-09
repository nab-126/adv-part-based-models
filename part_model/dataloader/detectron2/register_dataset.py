"""Register the ImageNet dataset in detectron2 format."""

import json
import logging
import os
from argparse import Namespace
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

from part_model.dataloader.imagenet_labels import IMAGENET_LABELS_TO_NAMES
from part_model.dataloader.util import get_metadata
from part_model.utils.image import get_seg_type
from part_model.utils.types import SamplesList

logger = logging.getLogger(__name__)


_DETECTRON2_DATASETS = ["imagenet", "part-imagenet", "paco"]


def _load_sem_seg(
    gt_root: Path,
    image_root: Path,
    image_ext: str = "JPEG",
    class_names: list[str] | None = None,
) -> SamplesList:
    """Load semantic segmentation datasets.

    (Modified from detectron2.data.datasets.load_sem_seg)
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext"
    extension are treated as ground truth annotations and all files under
    "image_root" with "image_ext" extension as input images. Ground truth and
    input images are matched using file paths relative to "gt_root" and
    "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root: full path to ground truth semantic segmentation files.
            Semantic segmentation annotations are stored as images with integer
            values in pixels that represent corresponding semantic labels.
            Example: "~/data/PartImageNet/PartSegmentations/All/".
        image_root: the directory where the input images are.
            Example: "~/data/PartImageNet/JPEGImages/".
        gt_ext: File extension for ground truth annotations. Defaults to "png".
        image_ext: File extension for input images. Defaults to "jpg".

    Returns:
        A list of dicts in detectron2 standard format without instance-level
        annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """
    logger.info("Loading images and labels from %s and %s", image_root, gt_root)
    input_files, gt_files, labels = [], [], []
    for i, label in enumerate(class_names):
        with open(str(gt_root / f"{label}.txt"), "r", encoding="utf-8") as file:
            filenames = sorted([f.strip() for f in file.readlines()])
        for filename in filenames:
            input_file = image_root / f"{filename}.{image_ext}"
            assert (
                input_file.exists()
            ), f"Input file {input_file} does not exist!"
            # Try both .png and .tif extensions for ground truth annotations
            gt_file_dir = gt_root / label
            name = filename.split("/", maxsplit=1)[-1]
            gt_file = gt_file_dir / f"{name}.png"
            if not gt_file.exists():
                gt_file = gt_file_dir / f"{name}.tif"
            assert (
                gt_file.exists()
            ), f"Ground truth file {gt_file} does not exist!"
            input_files.append(str(input_file))
            gt_files.append(str(gt_file))
            labels.append(i)

    assert len(input_files) == len(gt_files), (
        "Number of input images and ground truth annotations do not match "
        f"({len(input_files)} vs {len(gt_files)})!"
    )
    logger.info("Loaded %d images", len(input_files))

    dataset_dicts = []
    for img_path, gt_path, label in zip(input_files, gt_files, labels):
        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = gt_path
        record["obj_class"] = label
        dataset_dicts.append(record)

    return dataset_dicts


def _load_paco_dataset(
    img_folder: str, seg_folder: str, annotations: dict, obj_classes: list[str]
) -> SamplesList:
    """Load PACO dataset."""
    dataset_dicts = []
    for ann in annotations["images"]:
        filename = ann["file_name"]
        img_path = os.path.join(img_folder, filename)

        seg_filename = ann["seg_filename"]
        seg_path = os.path.join(seg_folder, seg_filename)

        supercategory = ann["supercategory"]
        label = obj_classes.index(supercategory)

        record = {}
        record["file_name"] = img_path
        record["sem_seg_file_name"] = seg_path
        record["obj_class"] = label
        dataset_dicts.append(record)
    return dataset_dicts


def _list_classes(seg_path: Path) -> list[str]:
    """List all classes (sub-directories) in a directory.

    Args:
        seg_path: Path to the directory containing the classes.

    Returns:
        A list of class names.
    """
    dirs = filter(lambda d: d.is_dir(), seg_path.iterdir())
    dirs = [d.name for d in dirs]
    return sorted(dirs)


def register_detectron_paco(args: Namespace) -> None:
    """Register PACO dataset.

    Args:
        args: Arguments from the command line.
    """
    for split in ["train", "val", "test"]:
        name = f"{args.dataset}_{split}"
        img_folder = os.path.join(
            args.data, "PartSegmentations", "All", split, "images"
        )
        seg_folder = os.path.join(
            args.data, "PartSegmentations", "All", split, "seg_masks"
        )
        ann_file_path = os.path.join(
            args.data, "PartSegmentations", "All", f"{split}.json"
        )
        with open(ann_file_path, "r", encoding="utf-8") as file:
            annotations = json.load(file)

        stuff_classes = []
        for ann in annotations["part_categories"]:
            if ann["name"] == "background":
                continue
            stuff_classes.append(ann["name"])

        metadata = get_metadata(args)
        colors = metadata.colormap[1 : len(stuff_classes) + 1]
        colors = (colors * 255).int().tolist()
        stuff_colors = dict(enumerate(colors))
        # Add background class if specified
        if args.seg_include_bg:
            stuff_classes.append("background")
            bg_color = (metadata.colormap[0] * 255).int().tolist()
            stuff_colors[len(stuff_classes) - 1] = bg_color
            bg_label = len(stuff_classes) - 1

        stuff_classes.append("ignore")
        ignore_color = [255, 255, 255]
        stuff_colors[len(stuff_classes) - 1] = ignore_color
        ignore_label = len(stuff_classes) - 1

        class_names = metadata.OBJ_CLASSES

        args.num_classes = len(class_names)
        args.input_dim = metadata.input_dim
        # Automatically set number of segmentation labels if not given
        if args.seg_labels == -1:
            # Subtract 1 because of ignore label
            args.seg_labels = len(stuff_classes) - 1

        DatasetCatalog.register(
            name,
            lambda: _load_paco_dataset(
                img_folder, seg_folder, annotations, class_names
            ),
        )
        MetadataCatalog.get(name).set(
            image_root=str(img_folder),
            sem_seg_root=str(seg_folder),
            stuff_classes=stuff_classes,
            stuff_colors=stuff_colors,
            ignore_label=ignore_label,
            model_ignore_label=ignore_label,
            thing_classes=class_names,
            bg_label=bg_label,
            model_bg_label=bg_label,
            model_num_classes=args.seg_labels,
        )




def register_detectron_imagenet(args: Namespace) -> None:
    """Register ImageNet or PartImageNet dataset.

    Args:
        args: Arguments from the command line.
    """
    img_dir = Path(args.data) / "JPEGImages"
    base_seg_label_dir = Path(args.seg_label_dir)
    class_names = _list_classes(base_seg_label_dir / "train")
    metadata = get_metadata(args)
    args.input_dim = metadata.input_dim
    args.num_classes = len(class_names)
    colormap = metadata.colormap
    # Include background class
    num_seg_labels = sum(metadata.classes_to_num_parts.values()) + 1

    # Automatically set number of segmentation labels if not given
    seg_type = get_seg_type(args)
    if args.seg_labels == -1:
        if seg_type == "group":
            args.seg_labels = metadata.part_to_metapart_mat.shape[1]
        else:
            args.seg_labels = num_seg_labels

    # Get class names and colors for visualization
    stuff_classes = []
    for class_id, num_parts in metadata.classes_to_num_parts.items():
        if args.dataset == "imagenet" and seg_type != "group":
            class_name = IMAGENET_LABELS_TO_NAMES[class_id]
        else:
            class_name = class_id
        stuff_classes.extend([f"{class_name}-{i}" for i in range(num_parts)])

    # Get color for visualization
    colors = colormap[1:num_seg_labels]
    colors = (colors * 255).int().tolist()
    stuff_colors = dict(enumerate(colors))

    # Add background class if specified
    if args.seg_include_bg:
        stuff_classes.append("background")
        bg_color = (colormap[0] * 255).int().tolist()
        stuff_colors[num_seg_labels - 1] = bg_color
    num_classes = len(stuff_classes)

    # Add label for ignored objects
    stuff_classes.append("ignore")
    ignore_color = [255, 255, 255]
    stuff_colors[len(stuff_classes) - 1] = ignore_color

    for split in ["train", "val", "test"]:
        gt_dir = base_seg_label_dir / split
        name = f"{args.dataset}_{split}"
        DatasetCatalog.register(
            name,
            lambda x=img_dir, y=gt_dir: _load_sem_seg(
                y, x, image_ext="JPEG", class_names=class_names
            ),
        )
        MetadataCatalog.get(name).set(
            image_root=str(img_dir),
            sem_seg_root=str(gt_dir),
            stuff_classes=stuff_classes,
            stuff_colors=stuff_colors,
            ignore_label=num_seg_labels,
            bg_label=num_seg_labels - 1,
            thing_classes=class_names,
            num_classes=num_classes,
            num_obj_classes=len(class_names),
            model_ignore_label=args.seg_labels,  # Meta
            model_bg_label=args.seg_labels - 1,  # Meta
            model_num_classes=args.seg_labels,  # Meta
        )


def register_detectron_dataset(args: Namespace) -> None:
    """Register Detectron2 dataset.

    Args:
        args: Arguments from the command line.
    """
    if args.dataset not in _DETECTRON2_DATASETS:
        raise ValueError(
            f"Unknown dataset: {args.dataset} for ImageNet. Must be in "
            f"{_DETECTRON2_DATASETS}!"
        )
    dataset_to_register_detectron = {
        "paco": register_detectron_paco,
        "part-imagenet": register_detectron_imagenet,
        "imagenet": register_detectron_imagenet,
    }
    return dataset_to_register_detectron[args.dataset](args)
