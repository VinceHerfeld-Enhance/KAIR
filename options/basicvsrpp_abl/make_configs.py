"""Generate the option JSONs of the 2-vs-4-branch BasicVSR++ study (one shared base, one delta per arm).

    python options/basicvsrpp_abl/make_configs.py      # rewrites options/basicvsrpp_abl/*.json

Edit BASE / ARMS here and regenerate rather than editing the JSONs by hand, so every arm keeps
the identical data, schedule and head. Protocol: BasicVSR++'s REDS recipe (30-frame clips,
256 GT patches, batch 8, Adam 1e-4, SpyNet lr x0.25 and frozen for 5k iters, Charbonnier),
shortened to TOTAL_ITER with the cosine period matched; comparisons are relative between arms.
"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_ITER = 150000
PILOT_ITER = 20000
ONE_ROUND = ["backward_1", "forward_1"]

BASE = {
    "task": None,
    "model": "vrt",
    "gpu_ids": [0, 1, 2, 3],
    "dist": True,
    "find_unused_parameters": False,
    "use_static_graph": True,
    "wandb_project": "BasicVSRpp_2branch",
    "scale": 4,
    "n_channels": 3,
    "path": {"root": "experiments/basicvsrpp_abl", "pretrained_netG": None, "pretrained_netE": None},
    "datasets": {
        "train": {
            "name": "train_dataset",
            "dataset_type": "VideoRecurrentTrainDataset",
            "dataroot_gt": "trainsets/REDS/train_sharp_with_val.lmdb",
            "dataroot_lq": "trainsets/REDS/train_sharp_bicubic_with_val.lmdb",
            "meta_info_file": "data/meta_info/meta_info_REDS_GT.txt",
            "filename_tmpl": "08d",
            "filename_ext": "png",
            "val_partition": "REDS4",
            "test_mode": False,
            "io_backend": {"type": "lmdb"},
            "num_frame": 30,
            "gt_size": 256,
            "interval_list": [1],
            "random_reverse": False,
            "use_hflip": True,
            "use_rot": True,
            "dataloader_shuffle": True,
            "dataloader_num_workers": 16,
            "dataloader_batch_size": 8,
        },
        "test": {
            "name": "test_dataset",
            "dataset_type": "VideoRecurrentTestDataset",
            "dataroot_gt": "testsets/REDS4/GT",
            "dataroot_lq": "testsets/REDS4/sharp_bicubic",
            "cache_data": True,
            "io_backend": {"type": "disk"},
            "num_frame": -1,
        },
    },
    "netG": {
        "net_type": "basicvsrpp_abl",
        "mid_channels": 64,
        "num_blocks": 7,
        "max_residue_magnitude": 10,
        "spynet_path": "model_zoo/basicvsrpp/spynet_sintel_final-3d2a1287.pth",
        "cpu_cache_length": 100,
        "branches": ["backward_1", "forward_1", "backward_2", "forward_2"],
        "align": "bilinear",
        "snap_grad": "mask_only",
        "flow_refine": "none",
        "flow_fit": "midpoint",
        "head_input": "all",
        "init_type": "default",  # keep basicsr inits (zero offset heads, pretrained SpyNet)
        "init_bn_type": "uniform",
        "init_gain": 0.2,
    },
    "train": {
        "G_lossfn_type": "charbonnier",
        "G_lossfn_weight": 1.0,
        "G_charbonnier_eps": 1e-12,
        "E_decay": 0,
        "G_optimizer_type": "adam",
        "G_optimizer_lr": 1e-4,
        "G_optimizer_betas": [0.9, 0.99],
        "G_optimizer_wd": 0,
        "G_optimizer_clipgrad": None,
        "G_optimizer_reuse": True,
        "fix_iter": 5000,
        "fix_lr_mul": 0.25,
        "fix_keys": ["spynet"],
        "total_iter": TOTAL_ITER,
        "G_scheduler_type": "CosineAnnealingWarmRestarts",
        "G_scheduler_periods": TOTAL_ITER,
        "G_scheduler_eta_min": 1e-7,
        "G_regularizer_orthstep": None,
        "G_regularizer_clipstep": None,
        "G_param_strict": True,
        "E_param_strict": True,
        "checkpoint_test": 10000,
        "checkpoint_save": 10000,
        "checkpoint_print": 200,
    },
    "val": {
        "save_img": False,
        "pad_seq": False,
        "flip_seq": False,
        "center_frame_only": False,
        "num_frame_testing": 0,
        "num_frame_overlapping": 2,
        "size_patch_testing": 0,
    },
}

# name -> netG overrides. Tier 1 answers the main question; tier 2 localises / disentangles it.
ARMS = {
    # tier 1
    "A_1round": {"branches": ONE_ROUND},
    "C_2round": {},
    "A5_1round_deep": {"branches": ONE_ROUND, "num_blocks": 23},  # FLOP+param matched to C (833 vs 830 GFLOPs/frame)
    "B1_snap": {"branches": ONE_ROUND, "align": "snap"},
    "B2_rqfp": {"branches": ONE_ROUND, "flow_refine": "rqfp"},
    "B3_snap_rqfp": {"branches": ONE_ROUND, "align": "snap", "flow_refine": "rqfp"},
    # tier 2
    "A3_3branch": {"branches": ONE_ROUND + ["backward_2"]},
    "D_2round_snap": {"align": "snap"},
    "B1_snap_ste": {"branches": ONE_ROUND, "align": "snap", "snap_grad": "ste"},
    # IART-style window: residual offsets pinned to 0 -> 3x3 raw taps around the flow, fraction-weighted.
    # vs B1_snap isolates "local window" from "multi-location gather" (Chan et al. AAAI'21 offset diversity).
    "B1_snap_window": {"branches": ONE_ROUND, "align": "snap", "max_residue_magnitude": 0},
    "B2_rqfp_spynet": {"branches": ONE_ROUND, "flow_refine": "rqfp_spynet"},
    "B2_rqfp_learned": {"branches": ONE_ROUND, "flow_refine": "rqfp_learned"},
}

# Short runs to check that snapped offsets still learn before committing the full B1/B3 runs.
PILOTS = {
    "pilot_snap_mask_only": {"branches": ONE_ROUND, "align": "snap"},
    "pilot_snap_ste": {"branches": ONE_ROUND, "align": "snap", "snap_grad": "ste"},
}


def build(name, net_overrides, total_iter):
    opt = copy.deepcopy(BASE)
    opt["task"] = f"bvpp_abl_{name}"
    opt["netG"].update(net_overrides)
    opt["train"]["total_iter"] = total_iter
    opt["train"]["G_scheduler_periods"] = total_iter
    if total_iter <= PILOT_ITER:
        opt["train"]["checkpoint_test"] = opt["train"]["checkpoint_save"] = 5000
    return opt


def main():
    jobs = [(n, o, TOTAL_ITER) for n, o in ARMS.items()] + [(n, o, PILOT_ITER) for n, o in PILOTS.items()]
    for name, overrides, total_iter in jobs:
        path = os.path.join(HERE, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(build(name, overrides, total_iter), fh, indent=2)
            fh.write("\n")
        print(path)


if __name__ == "__main__":
    main()
