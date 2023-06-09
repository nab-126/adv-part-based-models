"""Main script for both training and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
from torch.distributed.elastic.multiprocessing.errors import record
from torchmetrics import JaccardIndex as IoU
from torchvision.utils import save_image

from part_model.attacks.util import (
    setup_eval_attacker,
    setup_train_attacker,
    setup_val_attacker,
)
from part_model.dataloader import COLORMAP, load_dataset
from part_model.models import build_model
from part_model.utils import (
    AverageMeter,
    ProgressMeter,
    adjust_learning_rate,
    dist_barrier,
    get_compute_acc,
    get_rank,
    init_distributed_mode,
    is_main_process,
    pixel_accuracy,
    save_on_master,
)
from part_model.utils.argparse import get_args_parser
from part_model.utils.atta import ATTA
from part_model.utils.image import get_seg_type
from part_model.utils.loss import get_criterion

best_acc1 = 0


@record
def main(args):
    """Main function."""
    init_distributed_mode(args)

    global best_acc1

    # Fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loading code
    print("=> creating dataset")
    loaders = load_dataset(args)
    train_loader, train_sampler, val_loader, test_loader = loaders

    # Create model
    print("=> creating model")
    model, optimizer, scaler = build_model(args)
    cudnn.benchmark = True

    # Define loss function
    criterion, train_criterion = get_criterion(args)

    # Logging
    if is_main_process():
        logfile = open(os.path.join(args.output_dir, "log.txt"), "a")
        logfile.write(str(args) + "\n")
        logfile.flush()
        if args.wandb:
            wandb_id = os.path.split(args.output_dir)[-1]
            wandb.init(
                project="part-model", id=wandb_id, config=args, resume="allow"
            )
            print("wandb step:", wandb.run.step)

    eval_attack = setup_eval_attacker(args, model)
    no_attack = eval_attack[0][1]
    train_attack = setup_train_attacker(args, model)
    val_attack = setup_val_attacker(args, model)
    save_metrics = {
        "train": [],
        "test": [],
    }

    # Initialize ATTA training if specified
    atta: ATTA | None = None
    if args.adv_train == "atta":
        atta = ATTA(input_dim=args.input_dim, num_samples=args.num_train)

    print(args)

    if args.evaluate:
        if args.resume:
            load_path = args.resume
        else:
            load_path = f"{args.output_dir}/checkpoint_best.pt"
    else:
        print("=> beginning training")
        val_stats = {}
        for epoch in range(args.start_epoch, args.epochs):
            is_best = False
            if args.distributed:
                train_sampler.set_epoch(epoch)
            lr = adjust_learning_rate(optimizer, epoch, args)
            print(f"=> lr @ epoch {epoch}: {lr:.2e}")

            # Train for one epoch
            train_stats = _train(
                train_loader,
                model,
                train_criterion,
                train_attack,
                optimizer,
                scaler,
                epoch,
                args,
                atta=atta,
            )

            if (epoch + 1) % 2 == 0:
                val_stats = _validate(
                    val_loader, model, criterion, no_attack, args
                )
                clean_acc1, acc1 = val_stats["acc1"], None
                is_best = clean_acc1 > best_acc1
                if args.adv_train != "none":
                    val_stats = _validate(
                        val_loader, model, criterion, val_attack, args
                    )
                    acc1 = val_stats["acc1"]
                    is_best = acc1 > best_acc1 and clean_acc1 >= acc1

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc1": best_acc1,
                    "args": args,
                }

                if is_best:
                    print("=> Saving new best checkpoint")
                    save_on_master(save_dict, args.output_dir, is_best=True)
                    best_acc1 = (
                        max(clean_acc1, best_acc1)
                        if acc1 is None
                        else max(acc1, best_acc1)
                    )
                save_epoch = epoch + 1 if args.save_all_epochs else None
                save_on_master(
                    save_dict, args.output_dir, is_best=False, epoch=save_epoch
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
            }

            if is_main_process():
                save_metrics["train"].append(log_stats)
                if args.wandb:
                    wandb.log(log_stats)
                logfile.write(json.dumps(log_stats) + "\n")
                logfile.flush()

        # Compute stats of best model after training
        dist_barrier()
        load_path = f"{args.output_dir}/checkpoint_best.pt"

    print(f"=> loading checkpoint from {load_path}...")
    if args.gpu is None:
        checkpoint = torch.load(load_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = "cuda:{}".format(args.gpu)
        checkpoint = torch.load(load_path, map_location=loc)
    model.load_state_dict(checkpoint["state_dict"])

    # Running evaluation
    for attack in eval_attack:
        # Use DataParallel (not distributed) model for AutoAttack.
        # Otherwise, DDP model can get timeout or c10d failure.
        stats = _validate(test_loader, model, criterion, attack[1], args)
        print(f"=> {attack[0]}: {stats}")
        stats["attack"] = str(attack[0])
        dist_barrier()
        if is_main_process():
            save_metrics["test"].append(stats)
            if args.wandb:
                wandb.log(stats)
            logfile.write(json.dumps(stats) + "\n")

    if is_main_process():
        # Save metrics to pickle file if not exists else append
        pkl_path = os.path.join(args.output_dir, "metrics.pkl")
        if os.path.exists(pkl_path):
            metrics = pickle.load(open(pkl_path, "rb"))
            metrics.append(save_metrics)
        else:
            pickle.dump([save_metrics], open(pkl_path, "wb"))

        last_path = f"{args.output_dir}/checkpoint_last.pt"
        if os.path.exists(last_path):
            os.remove(last_path)
        logfile.close()


def _train(
    train_loader,
    model,
    criterion,
    attack,
    optimizer,
    scaler,
    epoch,
    args,
    atta: ATTA | None = None,
):
    use_seg: bool = get_seg_type(args) is not None
    seg_only: bool = "seg-only" in args.experiment
    tf_params_offset: int = 2 if not use_seg or seg_only else 3
    use_atta: bool = args.adv_train == "atta"

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Epoch: [{}]".format(epoch),
    )
    compute_acc = get_compute_acc(args)

    # Switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        segs: torch.Tensor | None = None
        if not use_seg or seg_only:
            # If training segmenter only, `targets` is segmentation mask
            images, targets = samples[:tf_params_offset]
        else:
            images, segs, targets = samples[:tf_params_offset]
            segs = segs.cuda(args.gpu, non_blocking=True)

        if use_atta:
            orig_images: torch.Tensor = images.clone()
            tf_params: list[torch.Tensor] = samples[tf_params_offset:]
            # Update image with saved perturbation
            images = atta.apply(images, tf_params)

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # Compute output
        with amp.autocast(enabled=not args.full_precision):
            if attack.use_mask:
                # Attack for part models where both class and segmentation
                # labels are used
                images = attack(images, targets, segs)
                if attack.dual_losses:
                    targets = torch.cat([targets, targets], dim=0)
                    segs = torch.cat([segs, segs], dim=0)
            else:
                # Attack for either classifier or segmenter alone
                images = attack(images, targets)

            if segs is None or seg_only:
                outputs = model(images)
                loss = criterion(outputs, targets)
            elif "groundtruth" in args.experiment:
                outputs = model(images, segs=segs)
                loss = criterion(outputs, targets)
            else:
                outputs = model(images, return_mask=True)
                loss = criterion(outputs, targets, segs)
                outputs = outputs[0]

            if args.adv_train in ("trades", "mat"):
                outputs = outputs[batch_size:]

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # Measure accuracy and record loss
        acc1 = compute_acc(outputs, targets)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)

        # Compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            if is_main_process() and args.wandb:
                wandb.log(
                    {
                        "acc": acc1.item(),
                        "loss": loss.item(),
                        "scaler": scaler.get_scale(),
                    }
                )
            progress.display(i)

        if use_atta:
            # Update saved perturbation in ATTA
            perturbation: torch.Tensor = images.cpu() - orig_images
            atta.update(perturbation, tf_params)

    progress.synchronize()
    return {
        "acc1": top1.avg,
        "loss": losses.avg,
        "lr": optimizer.param_groups[0]["lr"],
    }


def _validate(val_loader, model, criterion, attack, args):
    seg_only = "seg-only" in args.experiment
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    pacc = AverageMeter("PixelAcc", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    iou = AverageMeter("IoU", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, mem],
        prefix="Test: ",
    )
    compute_acc = get_compute_acc(args)
    compute_iou = IoU(args.seg_labels).cuda(args.gpu)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, samples in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        segs: torch.Tensor | None = None
        if len(samples) == 2:
            images, targets = samples
        elif seg_only:
            images, targets, _ = samples
        else:
            images, segs, targets = samples
            segs = segs.cuda(args.gpu, non_blocking=True)

        # DEBUG
        if args.debug:
            save_image(COLORMAP[segs].permute(0, 3, 1, 2), "gt.png")
            save_image(images, "test.png")

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        batch_size = images.size(0)

        # DEBUG: fixed clean segmentation masks
        if "clean" in args.experiment:
            model(images, clean=True)

        # compute output
        with torch.no_grad():
            if attack.use_mask:
                images = attack(images, targets, segs)
            else:
                images = attack(images, targets)

            # Need to duplicate segs and targets to match images expanded by
            # image corruption attack
            if images.shape[0] != targets.shape[0]:
                ratio = images.shape[0] // targets.shape[0]
                targets = targets.repeat(
                    (ratio,) + (1,) * (len(targets.shape) - 1)
                )
                if segs:
                    segs = segs.repeat((ratio,) + (1,) * (len(segs.shape) - 1))

            if segs is None or "normal" in args.experiment or seg_only:
                outputs = model(images)
            elif "groundtruth" in args.experiment:
                outputs = model(images, segs=segs)
                loss = criterion(outputs, targets)
            else:
                outputs, masks = model(images, return_mask=True)
                if "centroid" in args.experiment:
                    masks, _, _, _ = masks
                pixel_acc = pixel_accuracy(masks, segs)
                pacc.update(pixel_acc.item(), batch_size)
            loss = criterion(outputs, targets)

        # DEBUG
        # if args.debug and isinstance(attack, PGDAttackModule):
        if args.debug:
            save_image(
                COLORMAP[masks.argmax(1)].permute(0, 3, 1, 2),
                "pred_seg_clean.png",
            )
            print(targets == outputs.argmax(1))
            import pdb

            pdb.set_trace()

        # measure accuracy and record loss
        acc1 = compute_acc(outputs, targets)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        if seg_only:
            iou.update(compute_iou(outputs, targets).item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    progress.synchronize()
    print(f" * Acc@1 {top1.avg:.3f}")

    if pacc.count > 0:
        pacc.synchronize()
        print(f"Pixelwise accuracy: {pacc.avg:.4f}")
    if seg_only:
        iou.synchronize()
        print(f"IoU: {iou.avg:.4f}")
    return {"acc1": top1.avg, "loss": losses.avg, "pixel-acc": pacc.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Main script for part model (ATTA)", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
