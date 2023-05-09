"""Utility functions for dataloader."""

from argparse import Namespace
from typing import Any, Iterator

from detectron2.data import (
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from part_model.dataloader.detectron2.sem_seg_mapper import (
    CustomSemanticDatasetMapper,
)
from part_model.dataloader.detectron2.shuffle_inference_sampler import (
    ShuffleInferenceSampler,
)
from part_model.utils.image import get_seg_type


class WrappedDataLoader:
    """Wrapper to turn infinite data loader into epoch-based data loader."""

    def __init__(self, data_loader: Iterator, num_steps_per_epoch: int) -> None:
        """Initialize the wrapped data loader.

        This wrapper turns infinite data loader into epoch-based one.

        Args:
            data_loader: Base infinite data loader.
            num_steps_per_epoch: Number of steps in an epoch.
        """
        self.data_loader = iter(data_loader)
        self._length = num_steps_per_epoch
        self._index = 0  # Current index

    def __len__(self):
        """Get length of the wrapped data loader."""
        return self._length

    def __iter__(self):
        """Get iterator of the wrapped data loader."""
        return self

    def __next__(self):
        """Get next item of the wrapped data loader."""
        if self._index >= self._length:
            self._index = 0
            raise StopIteration
        self._index += 1
        return next(self.data_loader)


def get_metadata(dataset_name: str | Namespace) -> dict[str, Any]:
    """Get metadata of dataset.

    Args:
        dataset_name: Dataset name or arguments.

    Returns:
        dict: Metadata of dataset.
    """
    # Avoid circular import. pylint: disable=import-outside-toplevel
    from part_model.dataloader.cityscapes import CITYSCAPES
    from part_model.dataloader.imagenet import IMAGENET
    from part_model.dataloader.paco import PACO
    from part_model.dataloader.part_imagenet import PART_IMAGENET
    from part_model.dataloader.part_imagenet_bbox import PART_IMAGENET_BBOX
    from part_model.dataloader.part_imagenet_corrupt import (
        PART_IMAGENET_CORRUPT,
    )
    from part_model.dataloader.part_imagenet_geirhos import (
        PART_IMAGENET_GEIRHOS,
    )
    from part_model.dataloader.part_imagenet_mixed_next import (
        PART_IMAGENET_MIXED,
    )
    from part_model.dataloader.pascal_part import PASCAL_PART
    from part_model.dataloader.pascal_voc import PASCAL_VOC

    if not isinstance(dataset_name, str):
        dataset_name = dataset_name.dataset
    dataset_dict = {
        "cityscapes": CITYSCAPES,
        "paco": PACO,
        "pascal-part": PASCAL_PART,
        "pascal-voc": PASCAL_VOC,
        "part-imagenet": PART_IMAGENET,
        "part-imagenet-geirhos": PART_IMAGENET_GEIRHOS,
        "part-imagenet-mixed": PART_IMAGENET_MIXED,
        "part-imagenet-corrupt": PART_IMAGENET_CORRUPT,
        "part-imagenet-bbox": PART_IMAGENET_BBOX,
        "imagenet": IMAGENET,
    }
    return dataset_dict.get(dataset_name)


def load_dataset(args):
    """Load dataset.

    Args:
        args: Arguments.

    Returns:
        Dataloaders: train_loader, train_sampler, val_loader, test_loader.
    """
    # pylint: disable=import-outside-toplevel
    from part_model.dataloader.detectron2 import register_dataset
    from part_model.models.util import DETECTRON2_MODELS

    # Register the given dataset, used for detectron2 or visualizations
    register_dataset.register_detectron_dataset(args)

    metadata = get_metadata(args)
    if args.obj_det_arch == "dino":
        # DINO is a special case where part-imagenet-bbox dataset is used.
        # We might want a new interface or just change DINO to detectron2 in
        # the future.
        return metadata.loader(args)

    if (
        args.seg_arch in DETECTRON2_MODELS
        or args.obj_det_arch in DETECTRON2_MODELS
    ):
        # If object detection architecture is specified, we will use detectron2
        # format for the dataset. We can change this check in the future.
        # pylint: disable=import-outside-toplevel
        # from part_model.dataloader.detectron2 import register_dataset

        # Copy config over from args
        args.cfg.SOLVER.IMS_PER_BATCH = args.batch_size
        args.cfg.DATASETS.dataset = args.dataset
        args.cfg.DATASETS.TRAIN = [f"{args.dataset}_train"]
        args.cfg.DATASETS.TEST = [f"{args.dataset}_test"]
        args.cfg.DATALOADER.NUM_WORKERS = args.workers

        seg_label_map = None
        if get_seg_type(args) == "group":
            seg_label_map = metadata.part_to_metapart

        # Use config interface
        # pylint: disable=missing-kwoa,unexpected-keyword-arg,redundant-keyword-arg
        train_mapper = CustomSemanticDatasetMapper(
            args.cfg,
            is_train=True,
            seg_include_bg=args.seg_include_bg,
            color_jitter=args.color_jitter,
            seg_label_map=seg_label_map,
        )
        train_loader = build_detection_train_loader(
            args.cfg, mapper=train_mapper, aspect_ratio_grouping=False
        )
        test_mapper = CustomSemanticDatasetMapper(
            args.cfg,
            is_train=False,
            seg_include_bg=args.seg_include_bg,
            seg_label_map=seg_label_map,
        )
        val_len = len(DatasetCatalog.get(f"{args.dataset}_val"))
        val_loader = build_detection_test_loader(
            args.cfg,
            dataset_name=f"{args.dataset}_val",
            mapper=test_mapper,
            batch_size=args.batch_size,
            sampler=ShuffleInferenceSampler(val_len) if args.debug else None,
        )
        test_len = len(DatasetCatalog.get(f"{args.dataset}_test"))
        test_loader = build_detection_test_loader(
            args.cfg,
            dataset_name=f"{args.dataset}_test",
            mapper=test_mapper,
            batch_size=args.batch_size,
            sampler=ShuffleInferenceSampler(test_len) if args.debug else None,
        )

        # Wrap training data loader to get length and make it finite
        data_dicts = DatasetCatalog.get(f"{args.dataset}_train")
        num_batches = len(data_dicts) // (args.batch_size * args.world_size)
        if args.debug:
            num_batches = 2
        return (
            WrappedDataLoader(train_loader, num_batches),
            None,
            val_loader,
            test_loader,
        )

    if metadata is not None:
        return metadata.loader(args)
    raise ValueError(f"Unknown dataset: {args.dataset}!")
