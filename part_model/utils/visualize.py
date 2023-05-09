"""Visualization utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from torchvision.utils import save_image

from part_model.dataloader.util import get_metadata
from part_model.models.util import DETECTRON2_MODELS
from part_model.utils.types import BatchImages, BatchSegMasks

def visualize_sem_seg(
    args: argparse.Namespace,
    images: BatchImages,
    sem_seg_masks: BatchSegMasks,
    path: str = "figures/temp.png",
) -> None:
    """Creates a visualization of the semantic segmentation masks.

    Args:
        args: Command-line arguments.
        images: Batch of images.
        sem_seg_masks: Batch of semantic segmentation masks. Shape: [B, H, W].
        path: Path to save visualizations to. Defaults to "figures/temp.png".
    """
    metadata = MetadataCatalog.get(f"{args.dataset}_train")
    
    if not(args.seg_arch in DETECTRON2_MODELS or args.obj_det_arch in DETECTRON2_MODELS):
        ignore_label = metadata.ignore_label
        bg_class = metadata.bg_label
        sem_seg_masks -= 1
        sem_seg_masks[sem_seg_masks == -1] = bg_class
        sem_seg_masks[sem_seg_masks == -2] = ignore_label
        
    if metadata is None:
        meta = get_metadata(args)
        stuff_classes = []
        for class_name, num_parts in meta.classes_to_num_parts.items():
            stuff_classes.extend(
                [f"{class_name}-{i}" for i in range(num_parts)]
            )
        stuff_colors = meta.colormap[1 : len(stuff_classes) + 1]
        stuff_colors = (stuff_colors * 255).int().tolist()

        MetadataCatalog.get(f"{args.dataset}_train").set(
            stuff_classes=stuff_classes,
            stuff_colors=stuff_colors,
            ignore_label=len(stuff_classes),
        )
        metadata = MetadataCatalog.get(f"{args.dataset}_train")

    pl_path = Path(path)
    if not pl_path.parent.exists():
        pl_path.parent.mkdir(parents=True)

    images_cpu = images.permute(0, 2, 3, 1).cpu().numpy()
    vis = torch.zeros_like(images, device="cpu")
    for i, (image, seg) in enumerate(zip(images_cpu, sem_seg_masks)):
        visualizer = Visualizer(image, metadata)
        vis_output = visualizer.draw_sem_seg(seg.cpu())
        vis_output = torch.from_numpy(vis_output.get_image())
        vis[i] = vis_output.permute(2, 0, 1).float() / 255

    # Save all images in grid
    save_image(vis, path)