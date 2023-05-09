"""Utility for distributed training."""

from __future__ import annotations

import datetime
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):  # pylint: disable=redefined-builtin
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def dist_barrier():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, output_dir, is_best=True, epoch=None):
    if is_main_process():
        if is_best:
            path = f"{output_dir}/checkpoint_best.pt"
        else:
            path = f"{output_dir}/checkpoint_last.pt"

        torch.save(state, path)
        # Save to best model if not exist
        if not os.path.exists(f"{output_dir}/checkpoint_best.pt"):
            torch.save(state, f"{output_dir}/checkpoint_best.pt")

        if epoch is not None:
            path = f"{output_dir}/checkpoint_epoch{epoch}.pt"
            torch.save(state, path)


def init_distributed_mode(args) -> None:
    """Initialize distributed training mode."""
    if args.no_distributed:
        logger.info(
            "args.no_distributed is set to True. Forced no distributed mode."
        )
        args.distributed = False
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        logger.info("Using distributed mode...")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        logger.info("SLURM_PROCID found in environment variables.")
        logger.info("Using distributed mode with slurm...")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logger.info("Not using distributed mode.")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    logger.info("| distributed init (rank %d): %s", args.rank, args.dist_url)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(hours=12),
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    if get_rank() != 0:
        # Disable logging for non-master process to avoid logging slowdown
        logging.disable(logging.INFO)
