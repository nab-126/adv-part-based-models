from __future__ import annotations

import torch

from part_model.attacks.pgd import PGDAttack

EPS = 1e-6


class MATAttack(PGDAttack):
    def _project_l2(self, x, eps):
        dims = [-1,] + [
            1,
        ] * (x.ndim - 1)
        return x / (x.view(len(x), -1).norm(2, 1).view(dims) + EPS) * eps

    def _forward_l2(self, x, y):
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += self._project_l2(torch.randn_like(x_adv), self._eps)
            x_adv.clamp_(0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(x_adv, **self.forward_args)
                    loss = self._loss_fn(logits, y).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    delta = x_adv - x + self._project_l2(grads, self._step_size)
                    x_adv = x + self._project_l2(delta, self._eps)
                    # Clip perturbed inputs to image domain
                    x_adv.clamp_(0, 1)

            if self._num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self._loss_fn(self._core_model(x_adv), y).reshape(
                    worst_losses.shape
                )
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return x_adv_worst.detach()

    def _forward_linf(self, x, y):
        mode = self._core_model.training
        self._core_model.eval()

        # Initialize worst-case inputs
        x_adv_worst = x.clone().detach()
        worst_losses = torch.zeros(len(x), 1, 1, 1, device=x.device)

        # Repeat PGD for specified number of restarts
        for _ in range(self._num_restarts):
            x_adv = x.clone().detach()

            # Initialize adversarial inputs
            x_adv += torch.zeros_like(x_adv).uniform_(-self._eps, self._eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # Run PGD on inputs for specified number of steps
            for _ in range(self._num_steps):
                x_adv.requires_grad_()

                # Compute logits, loss, gradients
                with torch.enable_grad():
                    logits = self._core_model(x_adv, **self.forward_args)
                    loss = self._loss_fn(logits, y).mean()
                    grads = torch.autograd.grad(loss, x_adv)[0].detach()

                with torch.no_grad():
                    # Perform gradient update, project to norm ball
                    x_adv = x_adv.detach() + self._step_size * torch.sign(grads)
                    x_adv = torch.min(
                        torch.max(x_adv, x - self._eps), x + self._eps
                    )
                    # Clip perturbed inputs to image domain
                    x_adv = torch.clamp(x_adv, 0, 1)

            if self._num_restarts == 1:
                x_adv_worst = x_adv
            else:
                # Update worst-case inputs with itemized final losses
                fin_losses = self._loss_fn(self._core_model(x_adv), y).reshape(
                    worst_losses.shape
                )
                up_mask = (fin_losses >= worst_losses).float()
                x_adv_worst = x_adv * up_mask + x_adv_worst * (1 - up_mask)
                worst_losses = fin_losses * up_mask + worst_losses * (
                    1 - up_mask
                )

        # Return worst-case perturbed input logits
        self._core_model.train(mode)
        return torch.cat([x.detach(), x_adv_worst.detach()], dim=0)

    def forward(self, *args):
        if self._norm == "L2":
            return self._forward_l2(*args)
        return self._forward_linf(*args)
