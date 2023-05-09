"""Metric utilities."""

from __future__ import annotations

import math
from argparse import Namespace

import torch
import torch.distributed as dist


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        t = torch.tensor(
            [self.sum, self.count], dtype=torch.float64, device="cuda"
        )
        if dist.is_initialized():
            dist.barrier()
            dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def _adjust_learning_rate(
    args: Namespace,
    learning_rate: float,
    epoch: int,
    iteration: int | None = None,
) -> float:
    if iteration is not None and iteration < args.warmup_iters:
        learning_rate *= (iteration + 1) / args.warmup_iters
    elif epoch < args.warmup_epochs:
        # Linear warmup
        learning_rate *= (epoch + 1) / args.warmup_epochs
    if args.lr_schedule == "step":
        factor = sum(epoch >= step for step in args.lr_steps)
        learning_rate *= 0.1**factor
    elif args.lr_schedule == "cosine":
        learning_rate *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:
        raise NotImplementedError(f"Unknown lr schedule: {args.lr_schedule}")
    return learning_rate


def adjust_learning_rate(
    args: Namespace, optimizer, epoch: int, iteration: int | None = None
) -> float:
    """Decay the learning rate based on schedule."""
    lr = _adjust_learning_rate(args, args.lr, epoch, iteration=iteration)
    lr_backbone = _adjust_learning_rate(
        args, args.lr_backbone, epoch, iteration=iteration
    )
    # Apply new learning rate
    for param_group in optimizer.param_groups:
        if param_group.get("name", "non_backbone") == "non_backbone":
            param_group["lr"] = lr
        else:
            param_group["lr"] = lr_backbone
    return lr


def adjust_learning_rate_deeplabv3(
    optimizer, epoch, batch_idx, num_batches_per_epoch, args
):
    cur_iter = epoch * num_batches_per_epoch + batch_idx
    max_iter = args.epochs * num_batches_per_epoch
    lr = args.lr * (1 - float(cur_iter) / max_iter) ** 0.9
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[1]["lr"] = lr * args.last_mult


def adjust_epsilon(args: Namespace, attack, epoch: int):
    new_eps = args.epsilon
    if epoch < args.eps_warmup_epochs:
        new_eps *= (epoch + 1) / args.eps_warmup_epochs
        attack.set_eps(new_eps)
    return new_eps


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    # Return only top1
    return res[0]


@torch.no_grad()
def pixel_accuracy(output, target, is_logits=True):
    pred = output
    if is_logits:
        pred = output.argmax(1)
    acc = (pred == target).float().mean() * 100
    return acc


def get_compute_acc(args):
    if "seg-only" in args.experiment:
        return pixel_accuracy
    return accuracy
