"""Utility functions for logging/printing results."""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def log_result_dict_csv(
    args: argparse.Namespace,
    result_dict: dict[str, dict[str, float]],
    name: str = "Results in CSV format",
) -> str:
    """Print result dict into csv format."""
    # Format stats to be printed in csv format
    # Clean, PGD, AA accuracy (and clean, PGD, AA pixel accuracy or mAP)
    csv_stats = ["" for _ in range(6)]
    attack_to_idx = {
        "no_attack": 0,
        "pgd": 1,
        "aa": 2,
    }
    if args.obj_det_arch == "dino":
        stat_key = "map"
    elif args.experiment == "normal":
        stat_key = None
    else:
        stat_key = "pixel-acc"
    is_baseline = stat_key is None

    # Write stats to to list for csv
    for attack, stats in result_dict.items():
        stat_idx = attack_to_idx.get(attack)
        if stat_idx is None:
            continue
        csv_stats[stat_idx] = stats["acc1"]
        # Collect additional stats if exists, e.g., pixel-acc or mAP
        if stat_key is not None:
            csv_stats[stat_idx + 3] = stats[stat_key]
        else:
            csv_stats = csv_stats[:3]

    csv_log = ", ".join([f"{s:.1f}" if s else "" for s in csv_stats])
    # exp, arch, seg_arch, train, lr, wd, cclf, cseg, cdtt, noobj, cj, eps
    exp_name_tokens = [
        args.experiment.replace("-semi", ""),
        args.arch if is_baseline else args.seg_backbone,
        "" if is_baseline else args.seg_arch,
        f'{args.adv_train}{args.atk_steps if args.adv_train != "none" else ""}',
        args.lr,
        args.wd,
        "" if is_baseline else args.clf_const_trn,
        "" if is_baseline else args.seg_const_trn,
        args.d2_const_trn if args.is_detectron else "",
        args.cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT if args.is_detectron else "",
        args.color_jitter,
        args.pretrained,
        "/" if args.resume else "",
        f"{round(args.epsilon * 255)}/255" if args.adv_train != "none" else "",
    ]
    exp_name = ", ".join([str(s) for s in exp_name_tokens])
    csv_message = f"{name}: {exp_name}, {csv_log}"
    logger.info(csv_message)
    return csv_message
