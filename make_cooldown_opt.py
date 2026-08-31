#!/usr/bin/env python3
"""Derive a cooldown-phase option file from a base KAIR training config.

The problem this solves: you don't know in advance how many total iterations
you'll get before the deadline, so the base config's schedule (a long
G_scheduler_periods / a late fix_iter) is calibrated for a horizon you may
never reach. Rather than guessing a final total_iter up front, keep training
under the base config as long as you can, then branch a short "cooldown" run
from whatever checkpoint you're at when you actually have to stop.

Given a base opt json and the iteration you're branching from, this writes a
new opt json that:

1. Shortens total_iter to start_iter + cooldown_iters.
2. Appends a second LR-scheduler period, so the cosine anneal actually reaches
   eta_min by the new total_iter instead of barely moving along the original
   (much longer) curve -- e.g. resuming under an unchanged 1,000,000-length
   schedule for 50k more steps moves the LR by only ~5% of its remaining
   range, nowhere near eta_min.
3. Moves fix_iter to (approximately) the start of the cooldown, so whatever's
   gated by fix_keys (e.g. the flow submodule) gets fine-tuned for the
   cooldown window instead of staying frozen the whole time (KAIR's fix_iter
   check in model_elvsr.py is an exact `current_step == fix_iter`, so if
   fix_iter falls outside [start_iter, start_iter+cooldown_iters) it never
   fires again).

This script only edits schedule keys -- it never touches path.pretrained_*.
Resuming still goes through the base training script's normal auto-resume
(option.find_last_checkpoint scanning path.models), so make sure a checkpoint
at --start-iter actually exists under this config's path.models before you
launch; the script just warns if it doesn't find one, it doesn't fail.

Usage:
    python make_cooldown_opt.py \\
        options/stvsr/stvsr_flagship_clean_4gpu.json \\
        --start-iter 450000 --cooldown-iters 50000 \\
        --out options/stvsr/stvsr_flagship_clean_4gpu_cooldown.json
"""

import argparse
import copy
import json
import math
from pathlib import Path


def get_position_from_periods(iteration, cumulative_period):
    """Mirrors basicsr.models.lr_scheduler.get_position_from_periods exactly:
    index of the right-closest boundary, so the tie at a boundary itself
    belongs to the period ENDING there, not the one starting there."""
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i
    return len(cumulative_period) - 1


def cosine_restart_lr(step, periods, restart_weights, base_lr, eta_min):
    """Re-implements CosineAnnealingRestartLR.get_lr in plain Python (no torch
    import needed) so this script can run with any system python3."""
    cumulative = [sum(periods[: i + 1]) for i in range(len(periods))]
    idx = get_position_from_periods(step, cumulative)
    weight = restart_weights[idx]
    nearest_restart = 0 if idx == 0 else cumulative[idx - 1]
    period = periods[idx]
    return eta_min + weight * 0.5 * (base_lr - eta_min) * (
        1 + math.cos(math.pi * (step - nearest_restart) / period)
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_opt", help="Path to the source opt json (the run you're branching from).")
    ap.add_argument(
        "--start-iter",
        type=int,
        required=True,
        help="Absolute iteration of the checkpoint you're resuming from.",
    )
    ap.add_argument(
        "--cooldown-iters",
        type=int,
        required=True,
        help="Length of the final anneal-to-eta_min window.",
    )
    ap.add_argument("--out", required=True, help="Path to write the derived opt json.")
    ap.add_argument(
        "--restart-weight",
        type=float,
        default=None,
        help="Restart weight at the cooldown boundary. Default: auto-computed so LR is "
        "continuous with the base schedule at start-iter (no jump). Pass 1.0 for a full "
        "warm-restart back to the base LR instead.",
    )
    ap.add_argument(
        "--fix-iter-offset",
        type=int,
        default=0,
        help="fix_iter is set to start_iter + this offset: how far into the cooldown the "
        "fix_keys params (e.g. optical_flow_model) stay frozen before unlocking for the "
        "rest of the cooldown. Default 0: unlock immediately, so fix_keys gets the whole "
        "cooldown as its fine-tune window.",
    )
    args = ap.parse_args()

    base = json.loads(Path(args.base_opt).read_text())
    train = base["train"]

    base_lr = train["G_optimizer_lr"]
    eta_min = train["G_scheduler_eta_min"]
    orig_periods = train["G_scheduler_periods"]
    orig_weights = train.get("G_scheduler_restart_weights", [1] * len(orig_periods))
    if isinstance(orig_weights, (int, float)):
        orig_weights = [orig_weights] * len(orig_periods)

    if args.start_iter > sum(orig_periods):
        print(
            f"NOTE: start-iter ({args.start_iter}) is past the end of the base schedule "
            f"({sum(orig_periods)}); treating the base LR as pinned at eta_min there."
        )

    if args.restart_weight is not None:
        w = args.restart_weight
        print(f"Using explicit restart_weight={w} (no continuity check against the base schedule).")
    else:
        lr_at_start = cosine_restart_lr(args.start_iter, orig_periods, orig_weights, base_lr, eta_min)
        w = (lr_at_start - eta_min) / (base_lr - eta_min)
        print(
            f"Base schedule LR at step {args.start_iter}: {lr_at_start:.3e} "
            f"-> restart_weight={w:.4f} (continuous handoff, no LR jump)"
        )

    out = copy.deepcopy(base)
    out_train = out["train"]
    out_train["total_iter"] = args.start_iter + args.cooldown_iters
    out_train["G_scheduler_periods"] = [args.start_iter, args.cooldown_iters]
    out_train["G_scheduler_restart_weights"] = [1, w]

    if out_train.get("fix_iter", 0) and out_train.get("fix_keys"):
        new_fix_iter = args.start_iter + args.fix_iter_offset
        if not (args.start_iter <= new_fix_iter < out_train["total_iter"]):
            print(
                f"WARNING: fix_iter ({new_fix_iter}) is outside the cooldown window "
                f"[{args.start_iter}, {out_train['total_iter']}) -- it will never fire "
                f"(model_elvsr.py checks current_step == fix_iter exactly), so "
                f"{out_train['fix_keys']} would stay frozen the whole cooldown."
            )
        out_train["fix_iter"] = new_fix_iter
        print(
            f"fix_iter -> {new_fix_iter}: {out_train['fix_keys']} unlocks there and "
            f"fine-tunes for the rest of the cooldown."
        )
    elif out_train.get("fix_iter", 0):
        print("fix_iter is set but fix_keys is empty -- leaving fix_iter untouched.")

    # Best-effort check that a checkpoint actually exists at start_iter; warn, don't fail,
    # since you may be generating this ahead of the checkpoint landing on disk.
    try:
        models_dir = Path(out["path"]["root"]) / out["task"] / "models"
        expected = models_dir / f"{args.start_iter}_G.pth"
        if not expected.exists():
            print(
                f"WARNING: no checkpoint found at {expected} -- auto-resume "
                f"(option.find_last_checkpoint) will pick up whatever IS latest in "
                f"{models_dir}, which may not match the schedule this file assumes."
            )
    except KeyError:
        pass

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
