import sys
import math
import argparse
import time
import random
from models.ddp_model_elvsr import DDPModelELVSR
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

import os.path


"""
# --------------------------------------------
# training code for ELVSR
# Uses HuggingFace Accelerate for DDP.
# Launch with: accelerate launch ddp_train_elvsr.py --opt path/to/config.json
# or:          idr_accelerate ddp_train_elvsr.py --opt path/to/config.json
# --------------------------------------------
"""


def define_Model(opt):
    return DDPModelELVSR(opt)


def main(json_path="/home/vherfeld/Research/KAIR/options/elvsr/feature_v1.json"):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path, help="Path to option JSON file.")
    args = parser.parse_args()

    # ----------------------------------------
    # Initialize Accelerator (handles DDP automatically)
    # ----------------------------------------
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    opt = option.parse(args.opt, is_train=True)
    opt["dist"] = world_size > 1
    opt["rank"] = rank
    opt["world_size"] = world_size
    # Override num_gpu with accelerate's world_size for consistency
    opt["num_gpu"] = world_size

    if is_main:
        util.mkdirs((path for key, path in opt["path"].items() if "pretrained" not in key))

    # Wait for main process to create directories
    accelerator.wait_for_everyone()

    if is_main:
        wandb.init(
            project="KAIR_VideoSR",
            name=opt["task"] if "task" in opt else "run",
            config=opt,
        )

    # ----------------------------------------
    # update opt — find last checkpoint
    # ----------------------------------------
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

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if is_main:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger (only on main process)
    # ----------------------------------------
    logger = None
    if is_main:
        logger_name = "train"
        utils_logger.logger_info(logger_name, os.path.join(opt["path"]["log"], logger_name + ".log"))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed — use accelerate's set_seed for reproducibility across processes
    # ----------------------------------------
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    # Each process gets a different seed offset for data augmentation diversity,
    # but the model init seed is the same.
    if is_main:
        print("Base random seed: {}".format(seed))
    set_seed(seed + rank)

    """
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    """

    test_loader = None

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"]))
            if is_main:
                logger.info("Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_size))
            # Let Accelerate handle the DistributedSampler automatically
            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"] // world_size,
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"] // world_size,
                drop_last=True,
                pin_memory=True,
            )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
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
    if is_main:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # ----------------------------------------
    # Prepare model internals with Accelerator
    # ----------------------------------------
    # The model object from KAIR typically wraps its own netG, optimizer, scheduler.
    # We prepare them through accelerator so DDP, mixed-precision, etc. are handled.
    if hasattr(model, "netG") and hasattr(model, "G_optimizer") and hasattr(model, "schedulers"):
        model.netG, model.G_optimizer, train_loader = accelerator.prepare(model.netG, model.G_optimizer, train_loader)
        # If the model has a scheduler list, no need to prepare schedulers with accelerator
        # (they step based on iteration, not on optimizer step scaling).
    else:
        # Fallback: just prepare the dataloader
        train_loader = accelerator.prepare(train_loader)

    # Store accelerator reference on model so it can be used for backward if needed
    model.accelerator = accelerator

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """

    for epoch in range(1000000):  # keep running

        # Accelerate's dataloader handles setting the epoch for the internal DistributedSampler
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt["train"]["checkpoint_print"] == 0 and is_main:
                model.log_psnr()
                logs = model.current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.3e} ".format(k, v)
                logger.info(message)
                wandb.log({f"train/{k}": v for k, v in logs.items()}, step=current_step)
                wandb.log({"train/lr": model.current_learning_rate()}, step=current_step)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt["train"]["checkpoint_save"] == 0 and is_main:
                logger.info("Saving the model.")
                # Unwrap the model before saving so checkpoints are compatible with single-GPU loading
                _unwrap_and_save(model, accelerator, current_step)

            if opt["use_static_graph"] and (current_step == opt["train"]["fix_iter"] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                if is_main:
                    _unwrap_and_save(model, accelerator, current_step)
                current_step -= 1
                if is_main:
                    logger.info(
                        "Saving models ahead of time when changing the computation graph with "
                        "use_static_graph=True (we need it due to a bug with use_checkpoint=True "
                        "in distributed training). The training will be terminated by PyTorch in "
                        "the next iteration. Just resume training with the same .json config file."
                    )

            # -------------------------------
            # 6) testing (only on main process)
            # -------------------------------
            if current_step % opt["train"]["checkpoint_test"] == 0 and is_main:

                if test_loader is not None:
                    _run_validation(model, test_loader, opt, logger, epoch, current_step)

            # -------------------------------
            # 7) check termination
            # -------------------------------
            if current_step > opt["train"]["total_iter"]:
                if is_main:
                    logger.info("Finish training.")
                    _unwrap_and_save(model, accelerator, current_step)
                    wandb.finish()
                accelerator.wait_for_everyone()
                sys.exit()

    # Should never reach here
    if is_main:
        wandb.finish()


def _unwrap_and_save(model, accelerator, current_step):
    """
    Unwrap the DDP-wrapped network before saving so that the checkpoint
    is compatible with single-GPU / non-DDP loading.
    """
    # Temporarily replace netG with the unwrapped version for saving
    original_netG = model.netG
    model.netG = accelerator.unwrap_model(model.netG)
    model.save(current_step)
    model.netG = original_netG


def _run_validation(model, test_loader, opt, logger, epoch, current_step):
    """Run validation loop, compute metrics, log to wandb."""

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []

    has_gt = False

    for idx, test_data in enumerate(test_loader):
        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        output = visuals["E"]
        gt = visuals["H"] if "H" in visuals else None
        folder = test_data["folder"]

        test_results_folder = OrderedDict()
        test_results_folder["psnr"] = []
        test_results_folder["ssim"] = []
        test_results_folder["psnr_y"] = []
        test_results_folder["ssim_y"] = []

        for j in range(output.shape[0]):
            # -----------------------
            # save estimated image E
            # -----------------------
            img = output[j, ...].clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)

            if opt["val"]["save_img"]:
                save_dir = opt["path"]["images"]
                util.mkdir(save_dir)
                seq_ = os.path.basename(test_data["lq_path"][j][0]).split(".")[0]
                os.makedirs(f"{save_dir}/{folder[0]}", exist_ok=True)
                cv2.imwrite(f"{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png", img)

            # -----------------------
            # calculate PSNR / SSIM
            # -----------------------
            if gt is not None:
                has_gt = True
                img_gt = gt[j, ...].clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))
                img_gt = (img_gt * 255.0).round().astype(np.uint8)
                img_gt = np.squeeze(img_gt)

                test_results_folder["psnr"].append(util.calculate_psnr(img, img_gt, border=0))
                test_results_folder["ssim"].append(util.calculate_ssim(img, img_gt, border=0))
                if img_gt.ndim == 3:
                    img_y = util.bgr2ycbcr(img.astype(np.float32) / 255.0) * 255.0
                    img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.0) * 255.0
                    test_results_folder["psnr_y"].append(util.calculate_psnr(img_y, img_gt_y, border=0))
                    test_results_folder["ssim_y"].append(util.calculate_ssim(img_y, img_gt_y, border=0))
                else:
                    test_results_folder["psnr_y"] = test_results_folder["psnr"]
                    test_results_folder["ssim_y"] = test_results_folder["ssim"]

        if has_gt and len(test_results_folder["psnr"]) > 0:
            psnr = sum(test_results_folder["psnr"]) / len(test_results_folder["psnr"])
            ssim = sum(test_results_folder["ssim"]) / len(test_results_folder["ssim"])
            psnr_y = sum(test_results_folder["psnr_y"]) / len(test_results_folder["psnr_y"])
            ssim_y = sum(test_results_folder["ssim_y"]) / len(test_results_folder["ssim_y"])

            logger.info(
                "Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; "
                "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y)
            )
            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["psnr_y"].append(psnr_y)
            test_results["ssim_y"].append(ssim_y)
        else:
            logger.info("Testing {:20s}  ({:2d}/{})".format(folder[0], idx, len(test_loader)))

    # summarize psnr/ssim
    if has_gt and len(test_results["psnr"]) > 0:
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; "
            "PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}".format(epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y)
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


if __name__ == "__main__":
    main()
