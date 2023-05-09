"""Wrapped AutoAttack."""

from __future__ import annotations

from typing import Any, Callable

import torch

from autoattack_modified import AutoAttack
from part_model.attacks.base import AttackModule
from part_model.models.base_classifier import Classifier


class AutoAttackModule(AttackModule):
    """AutoAttack."""

    def __init__(
        self,
        attack_config: dict[str, Any],
        core_model: Classifier,
        loss_fn: Callable[..., torch.Tensor],
        norm: str = "Linf",
        eps: float = 8 / 255,
        num_classes: int = 10,
        **kwargs,
    ):
        """Initialize AutoAttackModule. For args, see AttackModule."""
        super().__init__(
            core_model,
            loss_fn,
            norm,
            eps,
            **kwargs,
        )
        _ = attack_config  # Unused
        self._num_classes: int = num_classes
        self._adversary = AutoAttack(
            self._core_model,
            norm=self._norm,
            eps=self._eps,
            version="standard",
            verbose=self._verbose,
            num_classes=self._num_classes,
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Run AutoAttack."""
        _ = kwargs  # Unused
        mode = self._core_model.training
        self._core_model.eval()
        x_adv = self._adversary.run_standard_evaluation(
            inputs, targets, bs=inputs.size(0)
        )
        self._core_model.train(mode)
        return x_adv
