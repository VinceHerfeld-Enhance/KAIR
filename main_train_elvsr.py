import sys
import os.path
import math
import argparse
import warnings

import time
import random
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import wandb


def _dataloader_worker_init(worker_id):
    """Disable OpenCV's internal thread pool inside each dataloader worker.

    Without this, every worker lets cv2.imdecode spawn its own threads, so 16
    workers oversubscribe the CPU and decode latency becomes erratic -> the
    prefetch queue drains unevenly and GPU utilisation oscillates. Re-applied
    here (not only at module import) so it holds under the 'spawn' start method.
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def _build_dataloader_kwargs(dataset_opt, runtime_opt=None, phase="train", distributed=False, num_gpu=1):
    runtime_opt = runtime_opt or {}
    loader_opt = runtime_opt.get("dataloader", {}) if isinstance(runtime_opt.get("dataloader", {}), dict) else {}

    if phase == "train":
        batch_size = (
            dataset_opt["dataloader_batch_size"] // num_gpu if distributed else dataset_opt["dataloader_batch_size"]
        )
        num_workers = (
            dataset_opt["dataloader_num_workers"] // num_gpu if distributed else dataset_opt["dataloader_num_workers"]
        )
        shuffle = False if distributed else dataset_opt["dataloader_shuffle"]
        default_prefetch = 4
    else:
        batch_size = dataset_opt.get("dataloader_batch_size", 1)
        num_workers = dataset_opt.get("dataloader_num_workers", loader_opt.get("val_num_workers", 2))
        shuffle = False
        default_prefetch = 2

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": phase == "train",
        "pin_memory": bool(loader_opt.get("pin_memory", True)),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_opt.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(loader_opt.get("prefetch_factor", default_prefetch))
        kwargs["worker_init_fn"] = _dataloader_worker_init

    return kwargs


def _setup_runtime_performance(opt, logger=None):
    """Enable safe runtime-level speed knobs for CUDA training."""
    train_opt = opt.get("train", {}) if isinstance(opt, dict) else {}
    perf_opt = train_opt.get("runtime", {}) if isinstance(train_opt.get("runtime", {}), dict) else {}

    if torch.cuda.is_available():
        cudnn_benchmark = bool(perf_opt.get("cudnn_benchmark", True))
        torch.backends.cudnn.benchmark = cudnn_benchmark

        allow_tf32 = bool(perf_opt.get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        matmul_precision = perf_opt.get("matmul_precision", "high")
        if matmul_precision in ("high", "medium", "highest"):
            torch.set_float32_matmul_precision(matmul_precision)

        msg = (
            f"Runtime perf: cudnn.benchmark={cudnn_benchmark}, "
            f"allow_tf32={allow_tf32}, matmul_precision={matmul_precision}"
        )
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

        loader_opt = perf_opt.get("dataloader", {}) if isinstance(perf_opt.get("dataloader", {}), dict) else {}
        if loader_opt:
            loader_msg = (
                "Runtime dataloader: "
                f"pin_memory={bool(loader_opt.get('pin_memory', True))}, "
                f"persistent_workers={bool(loader_opt.get('persistent_workers', True))}, "
                f"prefetch_factor={int(loader_opt.get('prefetch_factor', 4))}, "
                f"val_num_workers={int(loader_opt.get('val_num_workers', 2))}"
            )
            if logger is not None:
                logger.info(loader_msg)
            else:
                print(loader_msg)


def _extract_frame_number(filename):
    """Extract the numeric frame index from a filename like '00017.png' or 'frame_000017.png'."""
    import re

    stem = os.path.splitext(filename)[0]
    numbers = re.findall(r"\d+", stem)
    return int(numbers[-1]) if numbers else None


def _load_test_lr_indices(test_list_dir, folder_name, all_gt_paths):
    """Load LR frame indices from a test_list .txt file.

    Args:
        test_list_dir: Directory containing {folder}_im_list.txt files.
        folder_name: Video folder name (e.g. 'GOPR9635').
        all_gt_paths: List of all GT frame paths for this folder.

    Returns:
        lr_indices: LR frame indices into the full GT sequence.
        gt_start: First LR frame index (inclusive).
        gt_end: Last LR frame index (inclusive).
    """
    list_path = os.path.join(test_list_dir, f"{folder_name}_im_list.txt")
    with open(list_path, "r") as f:
        lr_filenames = [line.strip() for line in f if line.strip()]

    lr_frame_numbers = set(_extract_frame_number(fn) for fn in lr_filenames)
    lr_frame_numbers.discard(None)

    # Build mapping from frame number -> index in the full GT sequence
    gt_basenames = [os.path.basename(p) for p in all_gt_paths]
    gt_frame_numbers = [_extract_frame_number(name) for name in gt_basenames]
    lr_indices = sorted([i for i, num in enumerate(gt_frame_numbers) if num in lr_frame_numbers])

    assert len(lr_indices) == len(lr_filenames), (
        f"Could not match all LR frames for {folder_name}: " f"found {len(lr_indices)}/{len(lr_filenames)} in GT folder"
    )

    return lr_indices, lr_indices[0], lr_indices[-1]


"""
# --------------------------------------------
# training code for ELVSR
# Supports temporal super-resolution (TSR):
#   When netG.tsf > 1, the LR input is temporally subsampled
#   by keeping every tsf-th frame. The model reconstructs
#   all intermediate frames and loss is computed over the
#   full HR sequence.
# --------------------------------------------
"""


def main(json_path="/home/vherfeld/Research/KAIR/options/elvsr/feature_v1.json"):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path, help="Path to option JSON file.")
    parser.add_argument("--launcher", default="pytorch", help="job launcher")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist", default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt["dist"] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt["dist"]:
        init_dist("pytorch")
    opt["rank"], opt["world_size"] = get_dist_info()
    # assign GPU to each distributed process
    if opt["dist"]:
        torch.cuda.set_device(parser.parse_args().local_rank)
    if opt["rank"] == 0:
        util.mkdirs((path for key, path in opt["path"].items() if "pretrained" not in key))

    if opt["rank"] == 0:
        wandb.init(project="KAIR_VideoSR", name=opt["task"] if "task" in opt else "run", config=opt)
    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt["path"]["models"], net_type="G", pretrained_path=opt["path"]["pretrained_netG"]
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt["path"]["models"], net_type="E", pretrained_path=opt["path"]["pretrained_netE"]
    )
    opt["path"]["pretrained_netG"] = init_path_G
    opt["path"]["pretrained_netE"] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt["path"]["models"], net_type="optimizerG"
    )
    opt["path"]["pretrained_optimizerG"] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt["rank"] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt["rank"] == 0:
        logger_name = "train"
        utils_logger.logger_info(logger_name, os.path.join(opt["path"]["log"], logger_name + ".log"))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
        _setup_runtime_performance(opt, logger)
    else:
        _setup_runtime_performance(opt, None)

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # temporal scale factor (TSR)
    # ----------------------------------------
    tsf = int(opt["netG"].get("tsf", 1)) if opt["netG"] is not None else 1
    n_interp_samples = int(opt["train"]["n_interp_samples"]) if opt["train"]["n_interp_samples"] else None
    if tsf > 1 and opt["rank"] == 0:
        logger.info(f"Temporal super-resolution enabled: tsf={tsf}")
        if n_interp_samples is not None:
            assert 1 <= n_interp_samples < tsf, f"n_interp_samples ({n_interp_samples}) must be in [1, tsf-1={tsf - 1}]"
            logger.info(f"  Random temporal sampling: {n_interp_samples}/{tsf - 1} intermediate steps per gap")
        num_frame = opt["datasets"]["train"]["num_frame"]
        assert (num_frame - 1) % tsf == 0, (
            f"num_frame ({num_frame}) must satisfy (num_frame - 1) % tsf == 0 "
            f"for temporal SR with tsf={tsf}. "
            f"Valid values: {[k * tsf + 1 for k in range(1, 20)]}"
        )
        logger.info(
            f"  Dataset num_frame={num_frame} -> model input: {(num_frame - 1) // tsf + 1} frames, "
            f"output: {num_frame} frames"
        )

    """
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    """

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"]))
            if opt["rank"] == 0:
                logger.info("Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_size))
            train_loader_kwargs = _build_dataloader_kwargs(
                dataset_opt,
                opt.get("train", {}).get("runtime", {}),
                phase="train",
                distributed=opt["dist"],
                num_gpu=opt["num_gpu"],
            )
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set, shuffle=dataset_opt["dataloader_shuffle"], drop_last=True, seed=seed
                )
                train_loader = DataLoader(
                    train_set,
                    sampler=train_sampler,
                    **train_loader_kwargs,
                )
            else:
                train_loader = DataLoader(train_set, **train_loader_kwargs)

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader_kwargs = _build_dataloader_kwargs(
                dataset_opt,
                opt.get("train", {}).get("runtime", {}),
                phase="test",
            )
            test_loader = DataLoader(test_set, **test_loader_kwargs)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """

    model = define_Model(opt)
    model.init_train()
    if opt["rank"] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """
    saved_fixed = False  # avoid saving the same fixed test frames multiple times
    total_iter = opt["train"]["total_iter"]
    epoch = 0
    while current_step < total_iter:  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
            if current_step >= total_iter:
                break

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(min(current_step, total_iter))

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            # Only subsample LR here if the dataset did NOT already do it.
            # When the dataset config contains "tsf", the dataset returns
            # pre-subsampled LR frames; subsampling again would be wrong.
            dataset_handles_tsf = opt["datasets"]["train"].get("tsf", 1) not in (None, 1)
            if tsf > 1 and not dataset_handles_tsf:
                # Temporally subsample LR: keep every tsf-th frame
                # H stays full so loss covers all frames
                train_data["L"] = train_data["L"][:, ::tsf]

            # -------------------------------
            # 2b) temporal interpolation sub-step selection
            # -------------------------------
            # Preferred path: the dataset performs PER-SAMPLE sub-step sampling and
            # has already decoded only the needed GT frames (input-aligned + each
            # sample's sampled mids), returning the per-sample sub-steps as
            # "interp_k" [n_gaps, n_interp_samples]. H is already 1:1 with the
            # network output (structural order), so nothing is subset here.
            per_sample_interp = "interp_k" in train_data
            interp_steps_per_gap = None
            if not per_sample_interp and tsf > 1 and n_interp_samples is not None:
                # Legacy batch-shared path (configs that set n_interp_samples on the
                # train block but NOT on the dataset): one shared k per gap, with GT
                # subset here after the dataset decoded the full sequence.
                T_in = train_data["L"].size(1)
                n_gaps = T_in - 1
                interp_steps_per_gap = {}
                for gap_idx in range(n_gaps):
                    interp_steps_per_gap[gap_idx] = sorted(random.sample(range(1, tsf), n_interp_samples))

                gt_indices_set = set()
                for fi in range(T_in):
                    gt_indices_set.add(fi * tsf)  # input frame
                for gap_idx in range(n_gaps):
                    for k in interp_steps_per_gap[gap_idx]:
                        gt_indices_set.add(gap_idx * tsf + k)  # sampled mid
                gt_select = sorted(gt_indices_set)

                # Subset GT on CPU before sending to GPU
                train_data["H"] = train_data["H"][:, gt_select]
                # Store index mapping so RAFT can find the right frames in H_full
                train_data["H_full_indices"] = gt_select

            model.feed_data(train_data)

            # Set interp_steps and gt_indices AFTER feed_data (which resets them)
            if per_sample_interp:
                # [B, n_gaps, n_interp_samples] long tensor: per-sample sub-steps.
                # Output is 1:1 with H (structural order) — no gt_indices needed.
                model.interp_steps = train_data["interp_k"].to(model.device, non_blocking=True)
                model.gt_indices = None
            elif interp_steps_per_gap is not None:
                model.interp_steps = interp_steps_per_gap
                # With interp_steps active, model output is already 1:1 with gt_select — no gt_indices needed
                model.gt_indices = None
            else:
                model.interp_steps = None

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt["train"]["checkpoint_print"] == 0 and opt["rank"] == 0:
                model.log_psnr()  # calculate PSNR for the current batch, which is used for logging but not optimization
                logs = model.current_log()  # such as loss
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():  # merge log information into message
                    message += "{:s}: {:.3e} ".format(k, v)
                logger.info(message)
                if opt["rank"] == 0:
                    wandb_log = {f"train/{k}": v for k, v in logs.items()}
                    wandb_log["train/lr"] = model.current_learning_rate()
                    wandb.log(wandb_log, step=current_step)

            # -------------------------------
            # 4b) debug: save train frames
            # -------------------------------
            debug_every = opt["train"]["checkpoint_debug_frames"] if opt["train"] else None
            if debug_every and current_step % debug_every == 0 and opt["rank"] == 0:
                visuals = model.current_visuals()  # L: [T_in,C,H,W], E: [T_out,C,H,W], H: [T_out,C,H,W]
                debug_dir = os.path.join(opt["path"]["images"], "debug_train")
                for key in ["L", "E", "H"]:
                    if key not in visuals:
                        continue
                    frames = visuals[key].clamp_(0, 1).numpy()
                    out_dir = os.path.join(debug_dir, f"iter_{current_step:08d}", key)
                    os.makedirs(out_dir, exist_ok=True)
                    for fi in range(frames.shape[0]):
                        img = frames[fi]
                        if img.ndim == 3:
                            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
                        img = (img * 255.0).round().astype(np.uint8)
                        cv2.imwrite(os.path.join(out_dir, f"{fi:04d}.png"), img)
                logger.info(f"Debug train frames saved to {debug_dir}/iter_{current_step:08d}/")

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt["train"]["checkpoint_save"] == 0 and opt["rank"] == 0:
                logger.info("Saving the model.")
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt["train"]["checkpoint_test"] == 0 and opt["rank"] == 0:

                test_results = OrderedDict()
                test_results["psnr"] = []
                test_results["ssim"] = []
                test_results["psnr_y"] = []
                test_results["ssim_y"] = []

                test_list_dir = opt["datasets"]["test"].get("test_list_dir", None) if opt["datasets"]["test"] else None

                for idx, test_data in enumerate(test_loader):
                    folder = test_data["folder"]

                    if test_list_dir is not None:
                        # Adobe240-style: LQ and GT are already filtered by
                        # the dataset to only include the effective frames.
                        pass
                    elif tsf > 1:
                        test_data["L"] = test_data["L"][:, ::tsf]

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    output = visuals["E"]
                    gt = visuals["H"] if "H" in visuals else None

                    # For temporal SR the model may output fewer frames than
                    # the full GT sequence (when T-1 is not divisible by tsf).
                    # Trim GT to match output length.
                    if gt is not None and gt.shape[0] > output.shape[0]:
                        gt = gt[: output.shape[0]]

                    # -------------------------------
                    # debug: save L / E / H frames for first test video
                    # -------------------------------
                    if idx == 1 and opt["train"]["checkpoint_debug_frames"] and opt["rank"] == 0:
                        debug_val_dir = os.path.join(
                            opt["path"]["images"], "debug_val", f"iter_{current_step:08d}", folder[0]
                        )
                        lr_frames = visuals["L"]  # [T_lq, C, H, W]

                        modalities = [("E", output)]
                        if not saved_fixed:
                            modalities.insert(0, ("L", lr_frames))
                            modalities.append(("H", gt))
                            saved_fixed = True
                        for key, frames in modalities:
                            if frames is None:
                                continue
                            out_dir = os.path.join(debug_val_dir, key)
                            os.makedirs(out_dir, exist_ok=True)
                            for fi in range(frames.shape[0]):
                                fr = frames[fi].clamp_(0, 1).numpy()
                                if fr.ndim == 3:
                                    fr = np.transpose(fr[[2, 1, 0], :, :], (1, 2, 0))
                                fr = (fr * 255.0).round().astype(np.uint8)
                                cv2.imwrite(os.path.join(out_dir, f"{fi:05d}.png"), fr)
                        logger.info(
                            f"Debug val frames saved: {debug_val_dir}/ "
                            f"(L={lr_frames.shape[0]}, E={output.shape[0]}, H={gt.shape[0] if gt is not None else 0})"
                        )

                    test_results_folder = OrderedDict()
                    test_results_folder["psnr"] = []
                    test_results_folder["ssim"] = []
                    test_results_folder["psnr_y"] = []
                    test_results_folder["ssim_y"] = []

                    for i in range(output.shape[0]):
                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        img = output[i, ...].clamp_(0, 1).numpy()
                        if img.ndim == 3:
                            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                        if opt["val"]["save_img"]:
                            save_dir = opt["path"]["images"]
                            util.mkdir(save_dir)
                            seq_ = os.path.basename(test_data["lq_path"][i][0]).split(".")[0]
                            os.makedirs(f"{save_dir}/{folder[0]}", exist_ok=True)
                            cv2.imwrite(f"{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png", img)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        img_gt = gt[i, ...].clamp_(0, 1).numpy()
                        if img_gt.ndim == 3:
                            img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                        img_gt = np.squeeze(img_gt)

                        test_results_folder["psnr"].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder["ssim"].append(util.calculate_ssim(img, img_gt, border=0))
                        if img_gt.ndim == 3:  # RGB image
                            img = util.bgr2ycbcr(img.astype(np.float32) / 255.0) * 255.0
                            img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.0) * 255.0
                            test_results_folder["psnr_y"].append(util.calculate_psnr(img, img_gt, border=0))
                            test_results_folder["ssim_y"].append(util.calculate_ssim(img, img_gt, border=0))
                        else:
                            test_results_folder["psnr_y"] = test_results_folder["psnr"]
                            test_results_folder["ssim_y"] = test_results_folder["ssim"]

                    psnr = sum(test_results_folder["psnr"]) / len(test_results_folder["psnr"])
                    ssim = sum(test_results_folder["ssim"]) / len(test_results_folder["ssim"])
                    psnr_y = sum(test_results_folder["psnr_y"]) / len(test_results_folder["psnr_y"])
                    ssim_y = sum(test_results_folder["ssim_y"]) / len(test_results_folder["ssim_y"])

                    if gt is not None:
                        logger.info(
                            "Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; "
                            "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(
                                folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y
                            )
                        )
                        test_results["psnr"].append(psnr)
                        test_results["ssim"].append(ssim)
                        test_results["psnr_y"].append(psnr_y)
                        test_results["ssim_y"].append(ssim_y)
                    else:
                        logger.info("Testing {:20s}  ({:2d}/{})".format(folder[0], idx, len(test_loader)))

                # summarize psnr/ssim
                if gt is not None:
                    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
                    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
                    ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
                    ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
                    logger.info(
                        "<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; "
                        "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(
                            epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y
                        )
                    )
                    if opt["rank"] == 0:
                        wandb.log(
                            {
                                "val/PSNR": ave_psnr,
                                "val/SSIM": ave_ssim,
                                "val/PSNR_Y": ave_psnr_y,
                                "val/SSIM_Y": ave_ssim_y,
                            },
                            step=current_step,
                        )
        epoch += 1

    logger.info("Finish training.")
    model.save(current_step)

    # Final validation at last iteration
    if opt["rank"] == 0 and test_loader is not None:
        test_results = OrderedDict()
        test_results["psnr"] = []
        test_results["ssim"] = []
        test_results["psnr_y"] = []
        test_results["ssim_y"] = []

        test_list_dir = opt["datasets"]["test"].get("test_list_dir", None) if opt["datasets"]["test"] else None

        for idx, test_data in enumerate(test_loader):
            folder = test_data["folder"]

            if test_list_dir is not None:
                pass
            elif tsf > 1:
                test_data["L"] = test_data["L"][:, ::tsf]

            model.feed_data(test_data)
            model.test()

            visuals = model.current_visuals()
            output = visuals["E"]
            gt = visuals["H"] if "H" in visuals else None

            if gt is not None and gt.shape[0] > output.shape[0]:
                gt = gt[: output.shape[0]]

            test_results_folder = OrderedDict()
            test_results_folder["psnr"] = []
            test_results_folder["ssim"] = []
            test_results_folder["psnr_y"] = []
            test_results_folder["ssim_y"] = []

            for i in range(output.shape[0]):
                img = output[i, ...].clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
                img = (img * 255.0).round().astype(np.uint8)
                if opt["val"]["save_img"]:
                    save_dir = opt["path"]["images"]
                    util.mkdir(save_dir)
                    seq_ = os.path.basename(test_data["lq_path"][i][0]).split(".")[0]
                    os.makedirs(f"{save_dir}/{folder[0]}", exist_ok=True)
                    cv2.imwrite(f"{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png", img)

                if gt is not None:
                    img_gt = gt[i, ...].clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)
                    img_gt = np.squeeze(img_gt)

                    test_results_folder["psnr"].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder["ssim"].append(util.calculate_ssim(img, img_gt, border=0))
                    if img_gt.ndim == 3:
                        img = util.bgr2ycbcr(img.astype(np.float32) / 255.0) * 255.0
                        img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.0) * 255.0
                        test_results_folder["psnr_y"].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder["ssim_y"].append(util.calculate_ssim(img, img_gt, border=0))
                    else:
                        test_results_folder["psnr_y"] = test_results_folder["psnr"]
                        test_results_folder["ssim_y"] = test_results_folder["ssim"]

            if gt is not None and len(test_results_folder["psnr"]) > 0:
                psnr = sum(test_results_folder["psnr"]) / len(test_results_folder["psnr"])
                ssim = sum(test_results_folder["ssim"]) / len(test_results_folder["ssim"])
                psnr_y = sum(test_results_folder["psnr_y"]) / len(test_results_folder["psnr_y"])
                ssim_y = sum(test_results_folder["ssim_y"]) / len(test_results_folder["ssim_y"])
                logger.info(
                    "Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; "
                    "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(
                        folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y
                    )
                )
                test_results["psnr"].append(psnr)
                test_results["ssim"].append(ssim)
                test_results["psnr_y"].append(psnr_y)
                test_results["ssim_y"].append(ssim_y)
            else:
                logger.info("Testing {:20s}  ({:2d}/{})".format(folder[0], idx, len(test_loader)))

        if len(test_results["psnr"]) > 0:
            ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
            ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
            ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
            ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
            logger.info(
                "<Final, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; "
                "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y)
            )
            wandb.log(
                {
                    "val/PSNR": ave_psnr,
                    "val/SSIM": ave_ssim,
                    "val/PSNR_Y": ave_psnr_y,
                    "val/SSIM_Y": ave_ssim_y,
                },
                step=current_step,
            )

    if opt["rank"] == 0:
        wandb.finish()
    sys.exit()


if __name__ == "__main__":
    main()
