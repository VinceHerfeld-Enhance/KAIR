import numpy as np
import random
import cv2
import torch
from pathlib import Path
import torch.utils.data as data
from torchvision import transforms

import utils.utils_video as utils_video

# OpenCV spawns its own thread pool inside cv2.imdecode (used by imfrombytes).
# With multiple dataloader workers this oversubscribes the CPU: every worker
# fights for the same cores, decode latency becomes erratic, the prefetch queue
# drains unevenly and GPU utilisation sawtooths. Force single-threaded OpenCV so
# parallelism comes only from the dataloader workers. Set at import so that
# fork-started workers inherit it; worker_init_fn re-applies it for spawn.
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class VideoRecurrentTrainDataset(data.Dataset):
    """Video dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_XXX_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    720p_240fps_1 100 (720,1280,3)
    720p_240fps_3 100 (720,1280,3)
    ...

    Key examples: "720p_240fps_1/00000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get("scale", 4)
        self.gt_size = opt.get("gt_size", 256)
        self.gt_root = Path(opt["dataroot_gt"])
        # dataroot_lq is optional: arbitrary-scale mode synthesises the LR from the
        # GT crop and never opens an LQ database. Falling back to gt_root keeps the
        # attribute defined for the code paths that still reference it.
        self.lq_root = Path(opt["dataroot_lq"]) if opt.get("dataroot_lq") else self.gt_root
        self.filename_tmpl = opt.get("filename_tmpl", "08d")
        self.filename_tmpl_lq = opt.get("filename_tmpl_lq", self.filename_tmpl)
        self.filename_tmpl_gt = opt.get("filename_tmpl_gt", self.filename_tmpl)
        self.filename_prefix_lq = opt.get("filename_prefix_lq", "")
        self.filename_prefix_gt = opt.get("filename_prefix_gt", "")
        self.filename_ext = opt.get("filename_ext", "png")
        self.lq_frame_offset = opt.get("lq_frame_offset", 0)
        self.crop_min_variance = opt.get("crop_min_variance", 0.0)
        self.crop_max_retries = opt.get("crop_max_retries", 5)
        # raw_lmdb: frames are stored as raw uint8 (create_lmdb_raw.py). Enables
        # crop-before-decode: choose the crop box first, then read only the crop's
        # row-band from each frame value (no PNG decode, less I/O). See
        # utils_video.LmdbBackend.get_raw_crop.
        self.raw_lmdb = bool(opt.get("raw_lmdb", False))
        # Per-clip cache of LQ (h, w) for the crop-before-read path. Keyed by clip
        # so datasets whose clips differ in resolution (e.g. Adobe240) are handled
        # correctly; a single clip is always internally uniform.
        self._lq_hw_cache = {}
        self._gt_hw_cache = {}
        self.num_frame = opt["num_frame"]
        self.tsf = int(opt.get("tsf", 1))  # temporal scale factor: only load every tsf-th LQ frame
        # Per-sample temporal-interpolation supervision: each sample independently
        # samples n_interp_samples intermediate sub-steps per gap, and ONLY the
        # input-aligned + sampled intermediate GT frames are decoded (the rest are
        # never read from disk). Active only when tsf>1 and n_interp_samples is set.
        self.n_interp_samples = opt.get("n_interp_samples", None)
        self.per_sample_interp = self.tsf > 1 and self.n_interp_samples is not None
        if self.per_sample_interp:
            self.n_interp_samples = int(self.n_interp_samples)
            assert 1 <= self.n_interp_samples <= self.tsf - 1, (
                f"n_interp_samples ({self.n_interp_samples}) must be in [1, tsf-1] "
                f"(tsf={self.tsf})"
            )
            assert (self.num_frame - 1) % self.tsf == 0, (
                f"num_frame ({self.num_frame}) must satisfy (num_frame-1) % tsf == 0 "
                f"for per-sample temporal interpolation (tsf={self.tsf})"
            )

        # ---- Arbitrary-spatial-scale mode (``lq_patch`` set) ----
        # The LR patch size is the CONFIGURED quantity and the GT crop is inferred
        # per batch as ``round(lq_patch * s)``. That is the inverse of the legacy
        # geometry (``lq_patch = gt_size // scale``), whose floor division breaks
        # alignment at non-divisor scales: at gt_size=256, scale=3 it yields
        # lq_patch=85 while ``top_gt = top * 3`` indexes a 255-wide box, so the LQ
        # crop and the GT crop stop being the same content.
        #
        # No LQ is read from disk in this mode: the LR is synthesised from the GT
        # crop (on GPU, in the model's ``feed_data``), which is what MoTIF /
        # BF-STVSR / VideoINR / v3 all do for arbitrary-scale training. ``s``
        # arrives per item from the batch sampler (see ``MultiScaleBatchSampler``)
        # so the worker knows it BEFORE choosing a crop box — that is what keeps the
        # raw-lmdb crop-before-decode path usable, and it makes the read volume
        # shrink with the scale instead of always paying the s_max footprint.
        self.lq_patch = opt.get("lq_patch", None)
        self.multi_scale = self.lq_patch is not None
        if self.multi_scale:
            self.lq_patch = int(self.lq_patch)
            # The LR input is built from the GT frames at the input-aligned (anchor)
            # times, so every multiple of tsf must actually be among the decoded GT
            # frames. ``per_sample_interp`` guarantees that by construction
            # (``gt_pos`` starts from all ``g * tsf``); the ``sparse_gt_frames`` path
            # does not, so refuse rather than silently build the LR from the wrong
            # frames.
            if not self.per_sample_interp:
                raise ValueError(
                    "lq_patch (arbitrary-scale mode) requires the per-sample temporal "
                    "interpolation path, i.e. tsf > 1 and n_interp_samples set. Otherwise the "
                    "input-aligned GT frames the LR is derived from are not guaranteed to be loaded."
                )

        keys = []
        total_num_frames = []  # some clips may not have 100 frames
        start_frames = []  # some clips may not start from 00000
        with open(opt["meta_info_file"], "r") as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(" ")
                keys.extend(
                    [
                        f"{folder}/{i:{self.filename_tmpl}}"
                        for i in range(int(start_frame), int(start_frame) + int(frame_num))
                    ]
                )
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt["name"] == "REDS":
            if opt["val_partition"] == "REDS4":
                val_partition = ["000", "011", "015", "020"]
            elif opt["val_partition"] == "official":
                val_partition = [f"{v:03d}" for v in range(240, 270)]
            else:
                raise ValueError(
                    f'Wrong validation partition {opt["val_partition"]}.' f"Supported ones are ['official', 'REDS4']."
                )
        else:
            val_partition = []

        self.keys = []
        self.total_num_frames = []  # some clips may not have 100 frames
        self.start_frames = []
        if opt["test_mode"]:
            for i, v in zip(range(len(keys)), keys):
                if v.split("/")[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split("/")[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False
        if self.io_backend_opt["type"] == "lmdb":
            self.is_lmdb = True
            if self.multi_scale:
                # GT only: no LQ is ever read. Registering an "lq" client pointing at
                # the same directory would make lmdb.open() fail outright ("already
                # open in this process") when dataroot_lq is left equal to dataroot_gt.
                self.io_backend_opt["db_paths"] = [self.gt_root]
                self.io_backend_opt["client_keys"] = ["gt"]
            elif hasattr(self, "flow_root") and self.flow_root is not None:
                self.io_backend_opt["db_paths"] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt["client_keys"] = ["lq", "gt", "flow"]
            else:
                self.io_backend_opt["db_paths"] = [self.lq_root, self.gt_root]
                self.io_backend_opt["client_keys"] = ["lq", "gt"]

        if self.raw_lmdb and not self.is_lmdb:
            raise ValueError("raw_lmdb=True requires io_backend.type == 'lmdb'.")

        # temporal augmentation configs
        self.interval_list = opt.get("interval_list", [1])
        self.random_reverse = opt.get("random_reverse", False)
        interval_str = ",".join(str(x) for x in self.interval_list)
        print(f"Temporal augmentation interval list: [{interval_str}]; " f"random reverse is {self.random_reverse}.")

    def __getitem__(self, index):
        # In arbitrary-scale mode the sampler hands over ``(index, scale)`` instead
        # of a bare index, so the worker knows the scale before it reads anything.
        scale = None
        if isinstance(index, (tuple, list)):
            index, scale = index[0], float(index[1])
        if self.multi_scale and scale is None:
            raise ValueError(
                "lq_patch is set (arbitrary-scale mode) but this item arrived without a scale. "
                "Build the DataLoader with MultiScaleBatchSampler so every item in a batch "
                "shares one scale."
            )

        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split("/")  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # Determine which GT frame positions to load.
        interp_k = None
        if self.per_sample_interp:
            # Per-sample temporal interpolation: decode only the input-aligned GT
            # frames plus this sample's randomly sampled intermediate frames. Each
            # gap independently samples n_interp_samples sub-steps in [1, tsf-1].
            # GT layout is structural: per gap g we emit the input frame at g*tsf
            # followed by its sorted mids, so output frame positions are
            # deterministic given (gap, slot) and align 1:1 with the network output.
            n = len(neighbor_list)
            T_in = (n - 1) // self.tsf + 1
            n_gaps = T_in - 1
            interp_k = [
                sorted(random.sample(range(1, self.tsf), self.n_interp_samples)) for _ in range(n_gaps)
            ]
            gt_pos = [g * self.tsf for g in range(T_in)]  # input-aligned
            for g in range(n_gaps):
                for k in interp_k[g]:
                    gt_pos.append(g * self.tsf + k)  # sampled intermediate
            gt_indices = sorted(gt_pos)
            lq_indices = set(range(0, n, self.tsf))
        else:
            # sparse_gt_frames: total number of GT frames to supervise on per sample.
            # When set, always load the first and last GT frames plus random middle ones.
            # This avoids loading all (num_frame) HR frames when tsf is large.
            sparse_gt_frames = self.opt.get("sparse_gt_frames", None)
            if sparse_gt_frames is not None and sparse_gt_frames > 0 and len(neighbor_list) > sparse_gt_frames:
                n = len(neighbor_list)
                fixed_indices = [0, n - 1]
                n_random = max(0, sparse_gt_frames - 2)
                middle = [i for i in range(1, n - 1)]
                rand_middle = sorted(random.sample(middle, min(n_random, len(middle))))
                gt_indices = sorted(set(fixed_indices + rand_middle))
            else:
                gt_indices = list(range(len(neighbor_list)))

            # Determine which LQ frames to load (subsample by tsf for temporal SR)
            if self.tsf > 1:
                lq_indices = set(range(0, len(neighbor_list), self.tsf))
            else:
                lq_indices = set(range(len(neighbor_list)))

        img_lqs = []
        img_gts = []
        gt_indices_set = set(gt_indices)
        if self.multi_scale:
            # LR size fixed by config; GT crop inferred from this batch's scale.
            lq_patch = self.lq_patch
            gt_size = round(lq_patch * scale)
        else:
            gt_size = self.gt_size
            lq_patch = self.gt_size // self.scale

        if self.multi_scale and not self.raw_lmdb:
            # ---- HR-only decode-then-crop path (arbitrary spatial scale, no raw lmdb) ----
            # Slower than the crop-before-decode path below (reads/decodes whole
            # frames instead of just the crop bytes), but works against a plain
            # PNG lmdb/disk tree -- no raw-uint8 lmdb has to exist. ``imfrombytes``
            # already normalises to float32 [0,1] regardless of source format, so
            # this produces pixel-identical crops to the raw path for the same
            # (clip, frame, top, left, gt_size). No LQ key is touched either way.
            decoded = {}
            for t, neighbor in enumerate(neighbor_list):
                if t not in gt_indices_set:
                    continue
                gt_name = f"{self.filename_prefix_gt}{neighbor:{self.filename_tmpl_gt}}"
                img_gt_path = f"{clip_name}/{gt_name}" if self.is_lmdb else self.gt_root / clip_name / f"{gt_name}.{self.filename_ext}"
                img_bytes = self.file_client.get(img_gt_path, "gt")
                decoded[t] = utils_video.imfrombytes(img_bytes, float32=True)

            first_t = next(iter(decoded))
            h_gt, w_gt = decoded[first_t].shape[:2]
            if h_gt < gt_size or w_gt < gt_size:
                raise ValueError(
                    f"GT ({h_gt},{w_gt}) is smaller than the crop {gt_size} needed for "
                    f"lq_patch={lq_patch} at scale {scale:.4f} (clip {clip_name})."
                )

            n_tries = self.crop_max_retries if self.crop_min_variance > 0 else 0
            for _ in range(n_tries + 1):
                top = random.randint(0, h_gt - gt_size)
                left = random.randint(0, w_gt - gt_size)
                if self.crop_min_variance <= 0:
                    break
                cand = decoded[first_t][top : top + gt_size, left : left + gt_size]
                if cand.var() >= self.crop_min_variance:
                    break

            for t in sorted(decoded):
                img_gts.append(decoded[t][top : top + gt_size, left : left + gt_size])

        elif self.multi_scale:
            # ---- HR-only crop-before-decode path (arbitrary spatial scale, raw lmdb) ----
            # The crop box is chosen on the GT grid; there is only one grid now, so
            # the legacy ``top_gt = top * scale`` remap (and its rounding hazard)
            # is gone. No LQ key is touched at all.
            first_gt_frame = neighbor_list[0]
            first_gt_key = f"{clip_name}/{self.filename_prefix_gt}{first_gt_frame:{self.filename_tmpl_gt}}"
            h_gt, w_gt = self._gt_hw_cache.get(clip_name, (None, None))
            if h_gt is None:
                h_gt, w_gt = self.file_client.get_shape(first_gt_key, "gt")[:2]
                self._gt_hw_cache[clip_name] = (h_gt, w_gt)
            if h_gt < gt_size or w_gt < gt_size:
                raise ValueError(
                    f"GT ({h_gt},{w_gt}) is smaller than the crop {gt_size} needed for "
                    f"lq_patch={lq_patch} at scale {scale:.4f} (clip {clip_name})."
                )

            n_tries = self.crop_max_retries if self.crop_min_variance > 0 else 0
            for _ in range(n_tries + 1):
                top = random.randint(0, h_gt - gt_size)
                left = random.randint(0, w_gt - gt_size)
                if self.crop_min_variance <= 0:
                    break
                cand = self.file_client.get_raw_crop(first_gt_key, top, left, gt_size, gt_size, "gt")
                if (cand.astype(np.float32) / 255.0).var() >= self.crop_min_variance:
                    break

            for t, neighbor in enumerate(neighbor_list):
                if t not in gt_indices_set:
                    continue
                gt_name = f"{self.filename_prefix_gt}{neighbor:{self.filename_tmpl_gt}}"
                crop = self.file_client.get_raw_crop(
                    f"{clip_name}/{gt_name}", top, left, gt_size, gt_size, "gt"
                )
                img_gts.append(crop.astype(np.float32) / 255.0)

        elif self.raw_lmdb:
            # ---- crop-before-decode path (raw-uint8 lmdb) ----
            # Resolve LQ frame size once (assumes a uniform frame size across the
            # set, which holds for REDS). Only the 16-byte header is read.
            first_lq_frame = neighbor_list[0] + self.lq_frame_offset
            first_lq_key = f"{clip_name}/{self.filename_prefix_lq}{first_lq_frame:{self.filename_tmpl_lq}}"
            h_lq, w_lq = self._lq_hw_cache.get(clip_name, (None, None))
            if h_lq is None:
                h_lq, w_lq = self.file_client.get_shape(first_lq_key, "lq")[:2]
                self._lq_hw_cache[clip_name] = (h_lq, w_lq)
            if h_lq < lq_patch or w_lq < lq_patch:
                raise ValueError(f"LQ ({h_lq},{w_lq}) is smaller than patch size {lq_patch} for clip {clip_name}.")

            # Choose the crop box up front, re-rolling low-variance crops — mirrors
            # utils_video.paired_random_crop (variance measured on the first LQ
            # frame's candidate crop, itself read via crop-before-decode).
            n_tries = self.crop_max_retries if self.crop_min_variance > 0 else 0
            for _ in range(n_tries + 1):
                top = random.randint(0, h_lq - lq_patch)
                left = random.randint(0, w_lq - lq_patch)
                if self.crop_min_variance <= 0:
                    break
                cand = self.file_client.get_raw_crop(first_lq_key, top, left, lq_patch, lq_patch, "lq")
                if (cand.astype(np.float32) / 255.0).var() >= self.crop_min_variance:
                    break
            top_gt, left_gt = top * self.scale, left * self.scale

            for t, neighbor in enumerate(neighbor_list):
                lq_frame = neighbor + self.lq_frame_offset
                lq_name = f"{self.filename_prefix_lq}{lq_frame:{self.filename_tmpl_lq}}"
                gt_name = f"{self.filename_prefix_gt}{neighbor:{self.filename_tmpl_gt}}"
                if t in lq_indices:
                    crop = self.file_client.get_raw_crop(
                        f"{clip_name}/{lq_name}", top, left, lq_patch, lq_patch, "lq"
                    )
                    img_lqs.append(crop.astype(np.float32) / 255.0)
                if t in gt_indices:
                    crop = self.file_client.get_raw_crop(
                        f"{clip_name}/{gt_name}", top_gt, left_gt, self.gt_size, self.gt_size, "gt"
                    )
                    img_gts.append(crop.astype(np.float32) / 255.0)
        else:
            # ---- decode-then-crop path (png lmdb / disk) ----
            img_gt_path = None
            for t, neighbor in enumerate(neighbor_list):
                lq_frame = neighbor + self.lq_frame_offset
                lq_name = f"{self.filename_prefix_lq}{lq_frame:{self.filename_tmpl_lq}}"
                gt_name = f"{self.filename_prefix_gt}{neighbor:{self.filename_tmpl_gt}}"

                # get LQ only for temporally subsampled indices
                if t in lq_indices:
                    if self.is_lmdb:
                        img_lq_path = f"{clip_name}/{lq_name}"
                    else:
                        img_lq_path = self.lq_root / clip_name / f"{lq_name}.{self.filename_ext}"
                    img_bytes = self.file_client.get(img_lq_path, "lq")
                    img_lq = utils_video.imfrombytes(img_bytes, float32=True)
                    img_lqs.append(img_lq)

                # get GT only for selected indices
                if t in gt_indices:
                    if self.is_lmdb:
                        img_gt_path = f"{clip_name}/{gt_name}"
                    else:
                        img_gt_path = self.gt_root / clip_name / f"{gt_name}.{self.filename_ext}"
                    img_bytes = self.file_client.get(img_gt_path, "gt")
                    img_gt = utils_video.imfrombytes(img_bytes, float32=True)
                    img_gts.append(img_gt)

            # randomly crop
            img_gts, img_lqs = utils_video.paired_random_crop(
                img_gts,
                img_lqs,
                self.gt_size,
                self.scale,
                img_gt_path,
                min_variance=self.crop_min_variance,
                max_retries=self.crop_max_retries,
            )

        # augmentation - flip, rotate
        n_lq = len(img_lqs)
        n_gt = len(img_gts)
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt["use_hflip"], self.opt["use_rot"])

        img_results = utils_video.img2tensor(img_results)
        img_gts = torch.stack(img_results[n_lq:], dim=0)

        # img_lqs: (t_lq, c, h, w)  — temporally subsampled if tsf > 1
        # img_gts: (k, c, h, w)  where k <= num_frame (sparse) or k == num_frame (full)
        # key: str
        result = {"H": img_gts, "key": key}
        if self.multi_scale:
            # No LR on disk: the model synthesises it from the GT anchors (see
            # ModelELVSR.feed_data). Augmentation has already been applied to the GT,
            # and downscaling commutes with the flips/transpose ``augment`` uses on a
            # square crop, so deriving the LR afterwards is equivalent to augmenting
            # a precomputed LR.
            #
            # ``lq_sel`` are the positions WITHIN H of the input-aligned (anchor)
            # frames, i.e. the frames the LR is built from. H holds a sparse subset of
            # frame positions, so the anchors are not at a fixed stride inside it.
            result["lq_sel"] = torch.tensor(
                [j for j, p in enumerate(gt_indices) if p % self.tsf == 0], dtype=torch.long
            )
            # The realised scale, recomputed from the integer sizes rather than
            # carried over from the sampled float (v3's data.py:262 discipline): this
            # is the number that actually maps the LR grid onto the GT crop.
            result["gt_size"] = torch.tensor(gt_size, dtype=torch.long)
            result["lq_patch"] = torch.tensor(lq_patch, dtype=torch.long)
        else:
            result["L"] = torch.stack(img_results[:n_lq], dim=0)
        if self.per_sample_interp:
            # Per-sample sub-step indices, shape [n_gaps, n_interp_samples].
            # The model turns these into per-sample tau; H is already 1:1 with the
            # network output (structural order) so no gt_indices mapping is needed.
            result["interp_k"] = torch.tensor(interp_k, dtype=torch.long)
        else:
            n_out = (n_lq - 1) * self.tsf + 1 if self.tsf > 1 else n_lq
            if n_gt < n_out:
                result["gt_indices"] = torch.tensor(gt_indices, dtype=torch.long)
        return result

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainNonblindDenoisingDataset(VideoRecurrentTrainDataset):
    """Video dataset for training recurrent architectures in non-blind video denoising.

    Args:
        Same as VideoTestDataset.

    """

    def __init__(self, opt):
        super(VideoRecurrentTrainNonblindDenoisingDataset, self).__init__(opt)
        self.sigma_min = self.opt["sigma_min"] / 255.0
        self.sigma_max = self.opt["sigma_max"] / 255.0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split("/")  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f"{clip_name}/{neighbor:{self.filename_tmpl}}"
            else:
                img_gt_path = self.gt_root / clip_name / f"{neighbor:{self.filename_tmpl}}.{self.filename_ext}"

            # get GT
            img_bytes = self.file_client.get(img_gt_path, "gt")
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, _ = utils_video.paired_random_crop(img_gts, img_gts, self.gt_size, 1, img_gt_path)

        # augmentation - flip, rotate
        img_gts = utils_video.augment(img_gts, self.opt["use_hflip"], self.opt["use_rot"])

        img_gts = utils_video.img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)

        # we add noise in the network
        noise_level = torch.empty((1, 1, 1, 1)).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.normal(mean=0, std=noise_level.expand_as(img_gts))
        img_lqs = img_gts + noise

        t, _, h, w = img_lqs.shape
        img_lqs = torch.cat([img_lqs, noise_level.expand(t, 1, h, w)], 1)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {"L": img_lqs, "H": img_gts, "key": key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainVimeoDataset(data.Dataset):
    """Vimeo90K dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, separated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainVimeoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt["dataroot_gt"]), Path(opt["dataroot_lq"])
        self.temporal_scale = opt.get("temporal_scale", 1)

        with open(opt["meta_info_file"], "r") as fin:
            self.keys = [line.split(" ")[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False
        if self.io_backend_opt["type"] == "lmdb":
            self.is_lmdb = True
            self.io_backend_opt["db_paths"] = [self.lq_root, self.gt_root]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]

        # indices of input images
        self.neighbor_list = [i + (9 - opt["num_frame"]) // 2 for i in range(opt["num_frame"])][:: self.temporal_scale]

        # temporal augmentation configs
        self.random_reverse = opt["random_reverse"]
        print(f"Random reverse is {self.random_reverse}.")

        self.mirror_sequence = opt.get("mirror_sequence", False)
        self.pad_sequence = opt.get("pad_sequence", False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt["scale"]
        gt_size = self.opt["gt_size"]
        key = self.keys[index]
        clip, seq = key.split("/")  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f"{clip}/{seq}/im{neighbor}"
                img_gt_path = f"{clip}/{seq}/im{neighbor}"
            else:
                img_lq_path = self.lq_root / clip / seq / f"im{neighbor}.png"
                img_gt_path = self.gt_root / clip / seq / f"im{neighbor}.png"
            # LQ
            img_bytes = self.file_client.get(img_lq_path, "lq")
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            # GT
            img_bytes = self.file_client.get(img_gt_path, "gt")
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt["use_hflip"], self.opt["use_rot"])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)

        if self.mirror_sequence:  # mirror the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        elif self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            img_lqs = torch.cat([img_lqs, img_lqs[-1:, ...]], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:, ...]], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {"L": img_lqs, "H": img_gts, "key": key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainVimeoVFIDataset(VideoRecurrentTrainVimeoDataset):

    def __init__(self, opt):
        super(VideoRecurrentTrainVimeoVFIDataset, self).__init__(opt)
        self.color_jitter = self.opt.get("color_jitter", False)

        if self.color_jitter:
            self.transforms_color_jitter = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt["scale"]
        gt_size = self.opt["gt_size"]
        key = self.keys[index]
        clip, seq = key.split("/")  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f"{clip}/{seq}/im{neighbor}"
            else:
                img_lq_path = self.lq_root / clip / seq / f"im{neighbor}.png"
            # LQ
            img_bytes = self.file_client.get(img_lq_path, "lq")
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # GT
        if self.is_lmdb:
            img_gt_path = f"{clip}/{seq}/im4"
        else:
            img_gt_path = self.gt_root / clip / seq / "im4.png"

        img_bytes = self.file_client.get(img_gt_path, "gt")
        img_gt = utils_video.imfrombytes(img_bytes, float32=True)
        img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend([img_gts])
        img_results = utils_video.augment(img_lqs, self.opt["use_hflip"], self.opt["use_rot"])

        img_results = utils_video.img2tensor(img_results)
        img_results = torch.stack(img_results, dim=0)

        if self.color_jitter:  # same color_jitter for img_lqs and img_gts
            img_results = self.transforms_color_jitter(img_results)

        img_lqs = img_results[:-1, ...]
        img_gts = img_results[-1:, ...]

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {"L": img_lqs, "H": img_gts, "key": key}
