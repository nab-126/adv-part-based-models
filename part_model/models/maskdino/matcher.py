"""Modules to compute the matching cost and solve the corresponding LSAP.

DINO code:
Copyright (c) 2022 IDEA. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]

MaskDINO code:
Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by
Feng Li and Hao Zhang.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from MaskDINO.maskdino.utils.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: Predictions for each example. Float tensor of arbitrary shape.
        targets: A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs (0 for the negative
            class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """batch_sigmoid_ce_loss.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """Computes an assignment between targets and predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        cost_box: float = 0,
        cost_giou: float = 0,
        cost_clf: float = 0,
        num_points: int = 0,
        panoptic_on: bool = False,
    ) -> None:
        """Creates the matcher.

        Args:
            cost_class: This is the relative weight of the classification error
                in the matching cost
            cost_mask: This is the relative weight of the focal loss of the
                binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the
                binary mask in the matching cost
            cost_box: Weight for box loss.
            cost_giou: Weight for giou loss.
            cost_clf: Weight for classification loss.
            num_points: Number of points to sample from the mask.
            panoptic_on: Whether to use panoptic segmentation.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.cost_clf = cost_clf
        self.costs = ["cls", "box", "mask"]
        if cost_clf > 0:
            self.costs.append("clf")

        self.panoptic_on = panoptic_on

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points

    def _compute_cost_class(self, out_prob, tgt_ids):
        # TODO(design): Not sure why we use focal loss here when the training
        # loss is CE.
        # focal loss
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = (
            alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        )
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        return cost_class

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost=None):
        """More memory-friendly matching.

        Change cost to compute only certain loss in matching.
        """
        cost = cost or self.costs
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for i in range(batch_size):
            out_bbox = outputs["pred_boxes"][i]
            if "box" in cost:
                tgt_bbox = targets[i]["boxes"]
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )
            else:
                cost_bbox = torch.tensor(0).to(out_bbox)
                cost_giou = torch.tensor(0).to(out_bbox)

            # [num_queries, num_classes]
            out_prob = outputs["pred_logits"][i].sigmoid()
            tgt_ids = targets[i]["labels"]
            cost_class = self._compute_cost_class(out_prob, tgt_ids)

            if "clf" in cost:
                out_prob = outputs["pred_clf"][i].sigmoid()
                tgt_ids = targets[i]["obj_class"]
                cost_clf = self._compute_cost_class(out_prob, tgt_ids)
                assert cost_clf.shape == cost_bbox.shape
            else:
                cost_clf = torch.tensor(0).to(out_bbox)

            if "mask" in cost:
                # [num_queries, H_pred, W_pred]
                out_mask = outputs["pred_masks"][i]
                # gt masks are already padded when preparing target
                tgt_mask = targets[i]["masks"].to(out_mask)

                out_mask = out_mask[:, None]
                tgt_mask = tgt_mask[:, None]
                # all masks share the same set of points for efficient matching!
                point_coords = torch.rand(
                    1, self.num_points, 2, device=out_mask.device
                )
                # get gt labels
                tgt_mask = point_sample(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                out_mask = point_sample(
                    out_mask,
                    point_coords.repeat(out_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)

                with autocast(enabled=False):
                    out_mask = out_mask.float()
                    tgt_mask = tgt_mask.float()
                    # If there's no annotations
                    if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                        # Compute the focal loss between masks
                        cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                        # Compute the dice loss betwen masks
                        cost_dice = batch_dice_loss(out_mask, tgt_mask)
                    else:
                        cost_mask = batch_sigmoid_ce_loss_jit(
                            out_mask, tgt_mask
                        )
                        cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            else:
                cost_mask = torch.tensor(0).to(out_bbox)
                cost_dice = torch.tensor(0).to(out_bbox)

            # Final cost matrix
            if self.panoptic_on:
                isthing = tgt_ids < 80
                cost_bbox[:, ~isthing] = cost_bbox[:, isthing].mean()
                cost_giou[:, ~isthing] = cost_giou[:, isthing].mean()
                cost_bbox[cost_bbox.isnan()] = 0.0
                cost_giou[cost_giou.isnan()] = 0.0

            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                + self.cost_box * cost_bbox
                + self.cost_giou * cost_giou
                + self.cost_clf * cost_clf
            )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, cost=None):
        """Performs the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits": Classification logits with dim
                    [batch_size, num_queries, num_classes].
                "pred_masks": Predicted masks with dim
                    [batch_size, num_queries, H_pred, W_pred].

            targets: This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:
                "labels": Tensor of dim [num_target_boxes] (where
                    num_target_boxes is the number of ground-truth objects in
                    the target) containing the class labels.
                "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing
                    the target masks.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                index_i: indices of selected predictions (in order)
                index_j: indices of corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets, cost)

    def __repr__(self, _repr_indent=4):
        """Prints the matcher."""
        head = "Matcher " + self.__class__.__name__
        body = [
            f"cost_class: {self.cost_class}",
            f"cost_mask: {self.cost_mask}",
            f"cost_dice: {self.cost_dice}",
            f"cost_box: {self.cost_box}",
            f"cost_giou: {self.cost_giou}",
            f"cost_clf: {self.cost_clf}",
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
