import sys
import os.path
import math
import argparse
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
            if opt["dist"]:
                train_sampler = DistributedSampler(
                    train_set, shuffle=dataset_opt["dataloader_shuffle"], drop_last=True, seed=seed
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"] // opt["num_gpu"],
                    shuffle=False,
                    num_workers=dataset_opt["dataloader_num_workers"] // opt["num_gpu"],
                    drop_last=True,
                    pin_memory=True,
                    sampler=train_sampler,
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=dataset_opt["dataloader_batch_size"],
                    shuffle=dataset_opt["dataloader_shuffle"],
                    num_workers=dataset_opt["dataloader_num_workers"],
                    drop_last=True,
                    pin_memory=True,
                )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True
            )
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
            model.feed_data(train_data)

            # -------------------------------
            # 2b) random temporal step sampling
            # -------------------------------
            if tsf > 1 and n_interp_samples is not None:
                # Sample which intermediate steps to generate this iteration
                interp_steps = sorted(random.sample(range(1, tsf), n_interp_samples))
                model.interp_steps = interp_steps

                # Select matching GT frames: input frames + sampled intermediates
                T_in = model.L.size(1)
                gt_select = []
                for fi in range(T_in):
                    gt_select.append(fi * tsf)  # input frame position in full GT
                    if fi < T_in - 1:
                        for k in interp_steps:
                            gt_select.append(fi * tsf + k)
                model.H = model.H[:, gt_select]
                model.gt_indices = None  # output and GT are now 1:1 aligned

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
                    wandb.log({f"train/{k}": v for k, v in logs.items()}, step=current_step)
                    wandb.log({"train/lr": model.current_learning_rate()}, step=current_step)

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
    if opt["rank"] == 0:
        wandb.finish()
    sys.exit()


if __name__ == "__main__":
    main()
