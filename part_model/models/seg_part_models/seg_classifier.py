"""A wrapper classifier for segmentation models."""

from __future__ import annotations

import torch
from torchmetrics.functional import dice

from part_model.models.base_classifier import Classifier
from part_model.utils.types import (
    BatchImages,
    BatchLogitMasks,
    BatchSegMasks,
    OutputsDict,
    SamplesList,
)

_Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SegClassifier(Classifier):
    """Classifier interface with segmentation model."""

    def __init__(self, args, *sargs, **kwargs) -> None:
        """Initialize SegClassifier."""
        super().__init__(args, *sargs, **kwargs)
        self._num_seg_labels = args.seg_labels
        self._seg_include_bg = args.seg_include_bg
        assert self._num_seg_labels > 0
        bg_conf_thres = torch.zeros(self._num_seg_labels - 1)
        self.register_buffer("bg_conf_thres", bg_conf_thres, persistent=True)

    def _preprocess_detectron(self, inputs: SamplesList) -> _Inputs:
        """Preprocess batch of detectron2 samples.

        Args:
            inputs: List of samples in detectron2 format.

        Returns:
            Preprocessed batch of tensors.
        """
        images = torch.stack([inpt["image"] for inpt in inputs], dim=0).float()
        images /= 255

        # Get segmentation masks
        segs = torch.stack([inpt["sem_seg"] for inpt in inputs], dim=0)
        targets = torch.tensor(
            [inpt["obj_class"] for inpt in inputs],
            dtype=torch.long,
            device=self.device,
        )
        images = images.to(self.device, non_blocking=True)
        segs = segs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return images, targets, segs

    def _preprocess_torchvision(self, inputs: _Inputs) -> _Inputs:
        """Preprocess batch of tuple of samples for generic torchvision models.

        Args:
            inputs: Tuple of tensors.

        Returns:
            Preprocessed batch of tensors.
        """
        if len(inputs) != 3:
            raise ValueError(f"Expected 3 inputs, got {len(inputs)}!")
        prep_inputs = [t.to(self.device, non_blocking=True) for t in inputs]
        # Normalize images (assume as first input)
        if prep_inputs[0].dtype == torch.uint8:
            prep_inputs[0] = prep_inputs[0].float()
            prep_inputs[0] /= 255
        return prep_inputs

    def preprocess(self, inputs: _Inputs | SamplesList) -> _Inputs:
        """See Classifier.preprocess."""
        if self.is_detectron:
            return self._preprocess_detectron(inputs)
        return self._preprocess_torchvision(inputs)

    @torch.no_grad()
    def postprocess(self, outputs: OutputsDict) -> OutputsDict:
        """See Classifier.postprocess.

        Returns new output key "sem_seg_masks_t": segmentation mask with the
        threshold bg_conf_thres applied.
        """
        if not self.is_detectron or self._seg_include_bg:
            outputs["sem_seg_masks_t"] = outputs.get(
                "sem_seg_masks", outputs["sem_seg_logits"].argmax(1)
            )
            return outputs
        outputs["sem_seg_masks_t"] = self._apply_mask_conf(
            outputs["sem_seg_logits"]
        )
        return outputs

    @torch.no_grad()
    def _apply_mask_conf(
        self,
        sem_seg_logits: BatchLogitMasks,
        bg_conf_thres: float | torch.Tensor | None = None,
    ) -> BatchSegMasks:
        """Apply threshold on predicted segmentation mask.

        Args:
            sem_seg_logits: Predicted segmentation mask in logits.
            bg_conf_thres: Confidence threshold for background class. Any pixel
                with background-class confidence smaller than `bg_conf_thres`
                will be treated as foreground. If None, use self.bg_conf_thres.
                Defaults to None.

        Returns:
            Predicted hard-label segmentation mask.
        """
        # pylint: disable=no-member
        bg_conf_thres = bg_conf_thres or self.bg_conf_thres

        # Mask by threshold max foreground
        # seg_softmax = F.softmax(sem_seg_logits, dim=1)
        seg_probs = sem_seg_logits.sigmoid()
        num_classes = seg_probs.shape[1]  # No background class
        assert num_classes == self._num_seg_labels - 1
        # seg_masks_oh = F.one_hot(seg_masks.indices, num_classes=num_classes)
        # Get a mask of confidence score for each class
        # seg_vals = torch.einsum(
        #     "bhwp,bhw->bhwp", seg_masks_oh, seg_masks.values
        # )
        # fg_mask = (seg_vals > bg_conf_thres).sum(-1)
        # Set scores that are below threshold to 0
        seg_probs *= seg_probs > bg_conf_thres[None, :, None, None]
        fg_mask = (seg_probs.sum(1) > 0).long()
        seg_masks = seg_probs.max(1)
        seg_pred = fg_mask * seg_masks.indices + (1 - fg_mask) * num_classes
        return seg_pred

    @torch.no_grad()
    def compute_mask_conf(
        self, sem_seg_logits: BatchLogitMasks, seg_targets: BatchSegMasks
    ) -> torch.Tensor:
        """Update mask confidence and return best predicted masks.

        Only works with detectron2 models for now.

        Args:
            sem_seg_logits: Predicted segmentation mask in logits.
            seg_targets: Groundtruth segmentation mask.

        Returns:
            Predicted hard-label segmentation mask that achieves the best
            pixel-wise accuracy w.r.t. seg_targets.
        """
        if not self.is_detectron or self._seg_include_bg:
            return self.bg_conf_thres  # pylint: disable=no-member

        num_thres = 100
        device = sem_seg_logits.device
        # NOTE: We cannot just add zero logits here if we're taking softmax
        sem_seg_probs = torch.cat(
            [sem_seg_logits, torch.zeros_like(sem_seg_logits[:, :1])], dim=1
        )
        sem_seg_probs.sigmoid_()
        best_score = torch.zeros(self._num_seg_labels, device=device)
        best_thres = torch.zeros_like(best_score)
        for i in range(num_thres + 1):
            score = dice(
                sem_seg_probs,
                seg_targets,
                num_classes=self._num_seg_labels,
                threshold=i / num_thres,
                average=None,
                ignore_index=self._num_seg_labels - 1,
            )
            idx_better = score >= best_score
            best_score[idx_better] = score[idx_better]
            best_thres[idx_better] = i / num_thres

        return best_thres[:-1]

    def update_mask_conf(
        self,
        sem_seg_logits: BatchLogitMasks,
        seg_targets: BatchSegMasks,
        aggregate_mode: str | None = None,
        cur_num_samples: int | None = None,
    ) -> torch.Tensor:
        """Compute best background mask confidence and return its copy.

        This function updates `self.bg_mask_conf`.

        Args:
            sem_seg_logits: Predicted segmentation mask in logits.
            seg_targets: Groundtruth segmentation mask.
            aggregate_mode: Method to update `self.bg_mask_conf`. Options: None
                (does not update), "last" (last computed thres), "ema"
                (exponential moving average), "mean" (average over samples).
                Defaults to None.
            cur_num_samples: Current number of samples used to update with
                "mean" mode. Defaults to None.

        Returns:
            Updated `bg_mask_conf` copy.
        """
        ema_const = 0.9
        bg_mask_conf = self.compute_mask_conf(sem_seg_logits, seg_targets)
        if aggregate_mode is None:
            # Does not update self.bg_mask_conf by default
            return bg_mask_conf

        # pylint: disable=no-member
        if aggregate_mode == "last":  # or cur_bg_conf_thres is None:
            # Update with the computed confidence threshold
            self.bg_conf_thres[:] = bg_mask_conf
        elif aggregate_mode == "ema":
            # Expoential moving average update
            self.bg_conf_thres *= ema_const
            self.bg_conf_thres += (1 - ema_const) * bg_mask_conf
        else:
            # Mean update given number of samples seen
            batch_size = sem_seg_logits.shape[0]
            self.bg_conf_thres *= cur_num_samples
            self.bg_conf_thres += bg_mask_conf * batch_size
            self.bg_conf_thres /= cur_num_samples + batch_size
        return self.bg_conf_thres.clone()

    def forward(self, images: BatchImages, **kwargs) -> OutputsDict:
        """Default forward pass. Should be overriden by subclasses.

        Args:
            images: Batched input images with shape [B, 3, H, W].
            kwargs: Additional arguments for specific part-based models.

        Returns:
            Output dict containing the following keys:
                "class_logits": Predicted class logits. Shape: [B, num_classes].
                "sem_seg_logits": Predicted segmentation masks as logits.
                    Shape: [B, num_parts, H, W].
                "sem_seg_masks": Predicted hard-label segmentation masks.
                    Shape: [B, H, W].
                "losses": An optional loss dict from detectron2 models.
        """
        _ = kwargs  # Unused
        return_dict = self._init_return_dict()
        if self._normalize is not None:
            images = (images - self.mean) / self.std
        class_logits, sem_seg_logits = self._base_model(
            images, **self._forward_args
        )
        return_dict["class_logits"] = class_logits
        return_dict["sem_seg_logits"] = sem_seg_logits
        return_dict["sem_seg_masks"] = sem_seg_logits.detach().argmax(1)
        return return_dict
