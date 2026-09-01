import sys
import os.path
import math
import argparse
import re
import time

import random
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
from models.select_model import define_Model

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs


def _dataloader_worker_init(worker_id):
    """Disable OpenCV's internal thread pool inside each dataloader worker.

    Without this, every worker lets cv2.imdecode spawn its own threads, so the
    workers oversubscribe the CPU and decode latency becomes erratic -> the
    prefetch queue drains unevenly and GPU utilisation oscillates. Applied via
    worker_init_fn so it holds under the 'spawn' start method (Accelerate).
    """
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def _build_dataloader_kwargs(dataset_opt, runtime_opt=None, phase="train", world_size=1):
    """Build DataLoader kwargs, including the runtime perf knobs.

    Distribution is handled by Accelerate (it shards the prepared loader), so we
    keep ``shuffle`` from config and only divide the per-process batch/worker
    counts by ``world_size`` to keep the effective global batch size constant.
    """
    runtime_opt = runtime_opt or {}
    loader_opt = runtime_opt.get("dataloader", {}) if isinstance(runtime_opt.get("dataloader", {}), dict) else {}

    if phase == "train":
        batch_size = dataset_opt["dataloader_batch_size"]
        num_workers = dataset_opt["dataloader_num_workers"]
        if world_size > 1:
            batch_size = max(1, batch_size // world_size)
            num_workers = num_workers // world_size
        shuffle = dataset_opt["dataloader_shuffle"]
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

    # Arbitrary-spatial-scale training: one scale per optimizer step, carried with
    # the index so the worker knows it before it reads (see MultiScaleBatchSampler).
    # batch_size/shuffle/drop_last move onto the sampler; the loader must not also
    # set them or PyTorch raises.
    if phase == "train" and dataset_opt.get("lq_patch") is not None:
        from data.multi_scale_sampler import MultiScaleBatchSampler

        scale_range = dataset_opt.get("scale_range", [1.2, 4.0])
        kwargs["batch_sampler"] = MultiScaleBatchSampler(
            num_samples=dataset_opt["_num_samples"],
            batch_size=batch_size,
            scale_range=(scale_range[0], scale_range[1]),
            # One scale per optimizer step across ranks: Accelerate hands rank r the
            # batches at positions r, r+W, ..., so grouping by W makes the W batches
            # of one step share a scale.
            group_size=world_size,
            shuffle=shuffle,
            drop_last=True,
            seed=int(dataset_opt.get("scale_seed", 0)),
        )
        for key in ("batch_size", "shuffle", "drop_last"):
            kwargs.pop(key, None)

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


def _extract_frame_number(filename):
    """Extract the numeric frame index from a filename like '00017.png' or 'frame_000017.png'."""
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
# Uses HuggingFace Accelerate for (optional) DDP.
#   - single GPU:   python main_train_elvsr.py --opt path/to/config.json
#   - multi GPU:    accelerate launch main_train_elvsr.py --opt path/to/config.json
#                   idr_accelerate main_train_elvsr.py --opt path/to/config.json
# Supports temporal super-resolution (TSR):
#   When netG.tsf > 1, the LR input is temporally subsampled
#   by keeping every tsf-th frame. The model reconstructs
#   all intermediate frames and loss is computed over the
#   full HR sequence.
# --------------------------------------------
"""


def _accelerate_mixed_precision(opt):
    """Map the config AMP options to an Accelerate ``mixed_precision`` string.

    Under Accelerate, mixed precision / loss scaling is owned by the accelerator
    (the prepared model's forward runs under autocast and ``accelerator.backward``
    handles scaling). The model's internal AMP scaler is therefore disabled, so
    we translate ``use_amp``/``amp_dtype`` into the accelerator's setting instead.
    """
    train_opt = opt.get("train", {}) if isinstance(opt, dict) else {}
    if not bool(train_opt.get("use_amp", False)):
        return "no"
    dtype_name = str(train_opt.get("amp_dtype", "float16")).strip().lower()
    if dtype_name in ("bfloat16", "bf16"):
        return "bf16"
    return "fp16"


def main(json_path="/home/vherfeld/Research/KAIR/options/elvsr/feature_v1.json"):
    """
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default=json_path, help="Path to option JSON file.")
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help=(
            "For the first N iterations only, torch.cuda.synchronize() around "
            "optimize_parameters so the logged opt_time is true GPU time rather than "
            "launch-queue time. Off by default: a per-step synchronize is itself a "
            "cost this instrumentation exists to find."
        ),
    )
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    # ----------------------------------------
    # Initialize Accelerator (handles DDP automatically; single-process when
    # launched with plain `python`). Mixed precision is delegated to the
    # accelerator, so the model's internal AMP path is disabled below.
    # ----------------------------------------
    mixed_precision = _accelerate_mixed_precision(opt)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=opt.get("find_unused_parameters", True),
        # Align DDP bucket views with the grad memory layout. cudnn.benchmark can
        # emit 1x1-conv weight-grads in channels-last, mismatching the contiguous
        # bucket and triggering a per-step reconcile copy (+ a "Grad strides do
        # not match bucket view strides" warning). This removes both.
        gradient_as_bucket_view=True,
        # Wire through the option JSON's use_static_graph (previously read only by
        # model_base.py's native DDP path, never by this Accelerate-based one, so it
        # was a silent no-op here). Needed for modules whose parameters are reached by
        # reentrant backward passes (e.g. splatting.backward_gridsample's internal
        # torch.autograd.grad call), which otherwise trips DDP's "Expected to mark a
        # variable ready only once" error.
        static_graph=opt.get("use_static_graph", False),
    )
    accelerator = Accelerator(mixed_precision=mixed_precision, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Distribution is owned by Accelerate, not KAIR's native DDP wrapping.
    opt["dist"] = False
    opt["rank"] = rank
    opt["world_size"] = world_size
    opt["num_gpu"] = world_size
    # Disable the model's own AMP scaler/autocast — the accelerator provides it.
    if opt.get("train") is not None:
        opt["train"]["use_amp"] = False

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
        _setup_runtime_performance(opt, logger)
    else:
        _setup_runtime_performance(opt, None)

    # ----------------------------------------
    # seed — use accelerate's set_seed for reproducibility across processes
    # ----------------------------------------
    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    # Each process gets a different seed offset for data augmentation diversity.
    if is_main:
        print("Base random seed: {}".format(seed))
    set_seed(seed + rank)

    # ----------------------------------------
    # temporal scale factor (TSR)
    # ----------------------------------------
    tsf = int(opt["netG"].get("tsf", 1)) if opt["netG"] is not None else 1
    n_interp_samples = int(opt["train"]["n_interp_samples"]) if opt["train"]["n_interp_samples"] else None
    if tsf > 1 and is_main:
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
    # Step--2 (create dataloader)
    # ----------------------------------------
    """

    test_loader = None
    runtime_opt = opt.get("train", {}).get("runtime", {})

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"]))
            if is_main:
                logger.info("Number of train images: {:,d}, iters: {:,d}".format(len(train_set), train_size))
            # Accelerate shards the prepared loader across processes.
            # MultiScaleBatchSampler needs the dataset length up front.
            dataset_opt["_num_samples"] = len(train_set)
            train_loader_kwargs = _build_dataloader_kwargs(
                dataset_opt, runtime_opt, phase="train", world_size=world_size
            )
            train_loader = DataLoader(train_set, **train_loader_kwargs)
            if is_main and "batch_sampler" in train_loader_kwargs:
                sampler = train_loader_kwargs["batch_sampler"]
                logger.info(
                    f"  Arbitrary spatial scale: s ~ U({sampler.scale_lo}, {sampler.scale_hi}), "
                    f"one scale per optimizer step (group_size={sampler.group_size}); "
                    f"LR patch {dataset_opt['lq_patch']}, GT crop = round(lq_patch * s) "
                    f"in [{round(dataset_opt['lq_patch'] * sampler.scale_lo)}, "
                    f"{round(dataset_opt['lq_patch'] * sampler.scale_hi)}]"
                )

        elif phase == "test":
            test_set = define_Dataset(dataset_opt)
            test_loader_kwargs = _build_dataloader_kwargs(dataset_opt, runtime_opt, phase="test")
            test_loader = DataLoader(test_set, **test_loader_kwargs)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    """
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    """

    model = define_Model(opt)
    # Wire the accelerator into the model BEFORE init_train so loss modules and
    # data tensors are created on the correct per-process device.
    model.accelerator = accelerator
    model.device = device
    model.init_train()
    if is_main:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # ----------------------------------------
    # Prepare model internals with Accelerator (DDP, mixed precision, sharding).
    # The EMA network (if any) is moved manually — it is not optimized.
    # ----------------------------------------
    model.netG, model.G_optimizer, train_loader = accelerator.prepare(model.netG, model.G_optimizer, train_loader)
    if getattr(model, "netE", None) is not None:
        model.netE = model.netE.to(device)

    """
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    """
    saved_fixed = False  # avoid saving the same fixed test frames multiple times
    total_iter = opt["train"]["total_iter"]
    epoch = 0

    # ----------------------------------------
    # Step-time instrumentation
    # ----------------------------------------
    # There was no timing anywhere in this loop, so "training is slow" could not be
    # attributed: the STVSRNet profiler (VSR/scripts/profile_stvsr_step.py) measures
    # the *network*, explicitly excluding the RAFT teacher, the flow-supervision
    # losses and the dataloader. These three counters split the wall clock into
    # data_wait / opt_time / the Python remainder.
    #
    # Cost: two perf_counter calls per iteration and float adds. Nothing is read back
    # from the GPU here — the flush below reuses the existing checkpoint_print gate,
    # so no synchronisation is added to the hot path.
    timing = {"data_wait": 0.0, "step_time": 0.0, "opt_time": 0.0, "val_save": 0.0, "n": 0}
    profile_steps = max(0, int(args.profile_steps))
    body_end = time.perf_counter()

    while current_step < total_iter:  # keep running

        # Accelerate's dataloader handles setting the epoch for the internal sampler
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)

        body_end = time.perf_counter()  # epoch boundary: don't bill loader startup to data_wait

        for i, train_data in enumerate(train_loader):

            # Time spent blocked on the loader: everything between the end of the
            # previous body and the arrival of this batch. Non-zero here means the
            # prefetch queue drained, i.e. the step is data-bound.
            body_start = time.perf_counter()
            timing["data_wait"] += body_start - body_end

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
            # --profile-steps N: for the first N iterations, bracket with
            # synchronize() so opt_time is real GPU time instead of the time taken to
            # queue the launches. Off by default.
            sync_this_step = profile_steps and current_step <= profile_steps and torch.cuda.is_available()
            if sync_this_step:
                torch.cuda.synchronize()
            opt_start = time.perf_counter()

            model.optimize_parameters(current_step)

            if sync_this_step:
                torch.cuda.synchronize()
            timing["opt_time"] += time.perf_counter() - opt_start

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

            # The training step proper ends here. Everything below (debug frames,
            # checkpointing, validation) is periodic and billed separately, so a
            # save/validate iteration does not silently inflate step_time.
            core_end = time.perf_counter()
            timing["step_time"] += core_end - body_start
            timing["n"] += 1

            # -------------------------------
            # 4b) debug: save train frames
            # -------------------------------
            debug_every = opt["train"]["checkpoint_debug_frames"] if opt["train"] else None
            if debug_every and current_step % debug_every == 0 and is_main:
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
            if current_step % opt["train"]["checkpoint_save"] == 0 and is_main:
                logger.info("Saving the model.")
                # save_network unwraps the DDP-wrapped netG (get_bare_model), so
                # checkpoints stay compatible with single-GPU / non-DDP loading.
                model.save(current_step)

            # -------------------------------
            # 6) testing (only on main process)
            # -------------------------------
            if current_step % opt["train"]["checkpoint_test"] == 0 and is_main:
                if test_loader is not None:
                    saved_fixed = _run_validation(
                        model, test_loader, opt, logger, epoch, current_step, tsf, saved_fixed
                    )

            body_end = time.perf_counter()
            timing["val_save"] += body_end - core_end

            # -------------------------------
            # 4a) step-time / memory instrumentation
            # -------------------------------
            # Placed at the very END of the body on purpose. Flushing earlier read
            # timing["n"] before the step_time/n increment below, so opt_time had one
            # more sample than step_time and came out inflated by (n+1)/n -- which is
            # how a log line ended up reporting opt 1.375s inside a 1.371s step. Here
            # all five accumulators carry the same count.
            # The reset happens on EVERY rank so the running means stay comparable.
            if current_step % opt["train"]["checkpoint_print"] == 0:
                n = max(1, timing["n"])
                step_t = timing["step_time"] / n
                data_t = timing["data_wait"] / n
                opt_t = timing["opt_time"] / n
                # Checkpointing + validation, amortised over the window. Validation is
                # rank-0-only with no barrier, so its whole wall clock serialises into
                # training: the other ranks block at the next DDP allreduce.
                val_t = timing["val_save"] / n
                # Whatever is left is Python/logging overhead in the loop body.
                # NOT clamped: a negative value means the accumulators have drifted
                # out of step, and that should be visible rather than hidden.
                other_t = step_t - opt_t
                if is_main:
                    peak_alloc = peak_reserved = 0.0
                    if torch.cuda.is_available():
                        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
                        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
                    # gt_size is the arbitrary-scale sampler's realised HR crop for this
                    # step (round(lq_patch * scale)); absent in fixed-scale configs. It is
                    # a CPU tensor straight off the loader, so reading it costs no sync.
                    gt_side = None
                    if "gt_size" in train_data:
                        gt_side = int(train_data["gt_size"].reshape(-1)[0])
                    # Wall clock per iteration includes the amortised save/validation cost.
                    wall_t = step_t + data_t + val_t
                    iters_per_hour = 3600.0 / wall_t if wall_t > 0 else 0.0
                    logger.info(
                        "<iter:{:8,d}> wall: {:.3f}s = step {:.3f}s (opt {:.3f}s + other {:.3f}s) "
                        "+ data_wait {:.3f}s + save/val {:.3f}s | {:,.0f} it/h "
                        "| peak mem {:.2f}/{:.2f} GiB (alloc/reserved){}{}".format(
                            current_step,
                            wall_t,
                            step_t,
                            opt_t,
                            other_t,
                            data_t,
                            val_t,
                            iters_per_hour,
                            peak_alloc,
                            peak_reserved,
                            f" | gt_size {gt_side}" if gt_side is not None else "",
                            " | SYNCED" if profile_steps and current_step <= profile_steps else "",
                        )
                    )
                    wandb.log(
                        {
                            "perf/wall_time_s": wall_t,
                            "perf/step_time_s": step_t,
                            "perf/data_wait_s": data_t,
                            "perf/opt_time_s": opt_t,
                            "perf/other_time_s": other_t,
                            "perf/save_val_time_s": val_t,
                            "perf/iters_per_hour": iters_per_hour,
                            "perf/peak_mem_alloc_gib": peak_alloc,
                            "perf/peak_mem_reserved_gib": peak_reserved,
                            **({"perf/gt_size": gt_side} if gt_side is not None else {}),
                        },
                        step=current_step,
                    )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                timing = {"data_wait": 0.0, "step_time": 0.0, "opt_time": 0.0, "val_save": 0.0, "n": 0}


        epoch += 1

    if is_main:
        logger.info("Finish training.")
        model.save(current_step)

        # Final validation at last iteration
        if test_loader is not None:
            _run_validation(model, test_loader, opt, logger, epoch, current_step, tsf, saved_fixed)

        wandb.finish()
    accelerator.wait_for_everyone()
    sys.exit()


def _run_validation(model, test_loader, opt, logger, epoch, current_step, tsf, saved_fixed):
    """Run validation loop, compute metrics, log to wandb. Returns updated saved_fixed."""

    # SSIM is deliberately not computed here. PSNR-Y is the only training-time signal
    # used to steer a run; every other metric (SSIM, temporal consistency, ...) is
    # computed at benchmark time by ``vsr.benchmark``, which is the authoritative path.
    # numpy SSIM over full HR frames, run twice per frame (RGB + Y), was the bulk of
    # validation cost -- and validation is rank-0-only with no barrier, so its whole
    # wall clock serialises into training (the other ranks block at the next allreduce).
    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["psnr_y"] = []

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
        # debug: save L / E / H frames for one test video
        # -------------------------------
        if opt["val"]["checkpoint_debug_frames"]:
            debug_val_dir = os.path.join(opt["path"]["images"], "debug_val", f"iter_{current_step:08d}", folder[0])
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
        test_results_folder["psnr_y"] = []

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
            # calculate PSNR / PSNR-Y
            # -----------------------
            if gt is not None:
                img_gt = gt[j, ...].clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)
                img_gt = np.squeeze(img_gt)

                test_results_folder["psnr"].append(util.calculate_psnr(img, img_gt, border=0))
                if img_gt.ndim == 3:  # RGB image
                    img = util.bgr2ycbcr(img.astype(np.float32) / 255.0) * 255.0
                    img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.0) * 255.0
                    test_results_folder["psnr_y"].append(util.calculate_psnr(img, img_gt, border=0))
                else:
                    test_results_folder["psnr_y"] = test_results_folder["psnr"]

        if gt is not None and len(test_results_folder["psnr"]) > 0:
            psnr = sum(test_results_folder["psnr"]) / len(test_results_folder["psnr"])
            psnr_y = sum(test_results_folder["psnr_y"]) / len(test_results_folder["psnr_y"])

            logger.info(
                "Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; PSNR_Y: {:.2f} dB".format(
                    folder[0], idx, len(test_loader), psnr, psnr_y
                )
            )
            test_results["psnr"].append(psnr)
            test_results["psnr_y"].append(psnr_y)
        else:
            logger.info("Testing {:20s}  ({:2d}/{})".format(folder[0], idx, len(test_loader)))

    # summarize psnr
    if len(test_results["psnr"]) > 0:
        ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        logger.info(
            "<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; PSNR_Y: {:.2f} dB".format(
                epoch, current_step, ave_psnr, ave_psnr_y
            )
        )
        wandb.log(
            {
                "val/PSNR": ave_psnr,
                "val/PSNR_Y": ave_psnr_y,
            },
            step=current_step,
        )

    return saved_fixed


if __name__ == "__main__":
    main()
