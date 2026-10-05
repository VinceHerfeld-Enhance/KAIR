"""Build codec-compressed LR data for compression-robust STVSR training.

Two subcommands:

``train``  HR raw LMDB -> dense compressed LR raw LMDB, a drop-in ``dataroot_lq``.
``val``    HR frame folders + keyframe lists -> one LR folder per codec setting.

Why the train LMDB is dense and phase-interleaved
-------------------------------------------------
``VideoRecurrentTrainDataset`` starts a window at any frame ``s`` and reads LQ at
``s, s+tsf, s+2*tsf, ...`` (``data/dataset_video_train.py``, ``lq_indices``). Every
frame index can therefore be requested, but one window only ever touches frames of
a single phase ``s mod tsf``. So each clip is split into ``tsf`` phase streams
(frames ``p, p+tsf, p+2*tsf, ...``); each stream is encoded as its own
``fps``-rate video (default 30 = 240 / 8) and every decoded frame is written back
under its original key. The loader's existing tsf selection then always reads
frames that were coded together as one low-frame-rate stream: motion between
coded frames is the real 1/30 s motion, not the 1/240 s a dense encode would see.
No loader change is needed; point ``dataroot_lq`` at the output.

The input frame rate is fixed by the LMDB (240 / ``--tsf`` fps), not by the
model's temporal factor. The loader reads LR at ``s + tsf * interval * m``, so
any model tsf reuses this LMDB as long as ``tsf * interval`` equals the build's
``--tsf`` (8): tsf=8/interval=1, tsf=4/interval=2, tsf=2/interval=4. Validation
then needs GT at the output rate, written by ``val --gt-stride interval``.

Each (clip, phase) stream draws one setting: clean (prob ``--clean-prob``), or a
codec from ``--codecs`` and a tier (high/medium/low) with a CRF uniform in that
tier's range (``--tiers-json`` overrides ``DEFAULT_TIERS``). The draw is seeded by
(seed, clip, phase), so a rebuild is reproducible. Every stream's setting, bitrate
and PSNR against the clean LR is written to ``degradation_params.jsonl`` inside
the output LMDB folder.

Motion blur (``--exposures``, off by default): a real 30 fps camera integrates
light over its shutter time; a 240 fps source frame is near-instantaneous. Each
stream also draws an exposure length ``n`` (odd, <= tsf) and every LR keyframe
becomes the linear-light average of the ``n`` HR frames centred on it
(``vsr.utils.lr_synthesis``), before downscale and codec, as in the capture chain.
Centring keeps the sharp GT at the keyframe index as the mid-exposure frame.
``--exposures 1 3 5 7`` = 45/135/225/315 degree shutters at 30 fps. The exposure
uses its own RNG stream, so enabling it leaves every codec draw unchanged. In val
mode, prefix a setting with ``blur<n>+`` (e.g. ``blur5+x264:26``).

Capture + share chain (``--platform-prob``, off by default): real shared video
(WhatsApp, YouTube) was encoded by the phone at full resolution, then decoded,
downscaled by the platform's own scaler and re-encoded, sometimes more than once
(forwarding). A platform stream therefore runs: exposure -> x265 at HR (phone-like,
``--platform-hr-crf``, fast preset) -> downscale with a random non-MATLAB kernel
(``--platform-kernels``) -> ``--platform-codecs`` at LR (``--platform-crf``) -> with
``--platform-reshare-prob`` a second LR encode. It replaces that stream's single-encode
draw, from its own RNG stream (other draws unchanged). In val mode, e.g.
``hrx265:20+kbilinear+x264:32[+x264:34]``. VP9's CRF scale (0-63) differs from x264's.

Spatial downscale: MATLAB-faithful antialiased bicubic
(``vsr.utils.matlab_functions.imresize``), quantised as ``ToPILImage`` does
(``mul(255).byte()``, i.e. truncation). That is the operator the existing
``train_lr_raw.lmdb`` was built with (``VSR/scripts/downscale_videos.py`` matlab
mode), so the clean streams reproduce it. ``--reference-lr-lmdb`` checks that on
a few frames before the run.

Usage (from KAIR/, with the VSR package on the path)::

    PYTHONPATH=../VSR/src python scripts/data_preparation/create_compressed_lr_lmdb.py train \\
        --hr-lmdb   $D/Adobe240_train/train_hr_raw.lmdb \\
        --meta-info $D/Adobe240_train/meta_info_Adobe240_train_GT.txt \\
        --reference-lr-lmdb $D/Adobe240_train/train_lr_raw.lmdb \\
        --output-lmdb $D/Adobe240_train/train_lr_x264x265_mixed_raw.lmdb --workers 32

    PYTHONPATH=../VSR/src python scripts/data_preparation/create_compressed_lr_lmdb.py val \\
        --hr-dir $D/Adobe240_val/hr --list-dir $D/Adobe240_val/val_list --output-root $D/Adobe240_val

Smoke test first with ``--max-clips 2``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from collections import defaultdict
from multiprocessing import Pool
from os import path as osp

import cv2
import lmdb
import numpy as np
from PIL import Image

kair_root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
sys.path.append(kair_root)
sys.path.append(osp.dirname(osp.abspath(__file__)))

from create_lmdb_raw import HEADER_BYTES, pack_raw  # noqa: E402
from utils.utils_video import RAW_LMDB_MAGIC  # noqa: E402

try:
    import torch

    from vsr.utils.codec import CODECS, chain_roundtrip, codec_roundtrip, lossless_check, psnr_uint8, stream_roundtrip
    from vsr.utils.lr_synthesis import (
        DOWNSCALE_KERNELS, downscale_kernel_to_uint8, downscale_to_uint8, exposure_window, integrate_exposure,
        synthesize_lr, to_float_chw,
    )
except ImportError as e:  # pragma: no cover - environment problem, say how to fix it
    raise SystemExit(f"{e}\nRun with the VSR package on the path: PYTHONPATH=<code>/VSR/src") from e

# CRF ranges per codec and tier. x265 sits ~3 CRF above x264 for similar quality.
# Recalibrate from degradation_params.jsonl (kb/s, PSNR) once the RAFT study is in.
DEFAULT_TIERS = {
    "x264": {"high": [18, 23], "medium": [24, 29], "low": [30, 36]},
    "x265": {"high": [21, 26], "medium": [27, 32], "low": [33, 39]},
    "av1": {"high": [25, 32], "medium": [33, 42], "low": [43, 52]},
}

# Held-out-friendly default val settings: "clean", or "+"-joined stages: an optional leading
# "blur<n>" (n-frame exposure) followed by codec:crf stages.
DEFAULT_VAL_SETTINGS = [
    "clean", "x264:26", "x264:33", "x265:30", "av1:38", "x264:23+x265:30", "blur5", "blur5+x264:26",
]


# ----------------------------------------------------------------------------- helpers
def unpack_raw(value: bytes) -> np.ndarray:
    """LMDB value -> HWC BGR uint8. Raw-uint8 (``create_lmdb_raw.py``) or PNG-encoded (``create_lmdb.py``)."""
    if value[:4] == RAW_LMDB_MAGIC:
        h, w, c = np.frombuffer(value[4:HEADER_BYTES], dtype="<u4")
        return np.frombuffer(value, dtype=np.uint8, offset=HEADER_BYTES).reshape(h, w, c)
    img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("LMDB value is neither raw-uint8 (RAW1) nor a decodable image")
    return img


def pack_value(bgr: np.ndarray, fmt: str) -> bytes:
    """HWC BGR uint8 -> LMDB value: ``raw`` (loader ``raw_lmdb: true``) or ``png`` (``raw_lmdb: false``)."""
    if fmt == "raw":
        return pack_raw(bgr)[0]
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # lossless; level 1 as utils_lmdb
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def downscale_rgb(hr_rgb: np.ndarray, scale: int) -> np.ndarray:
    """uint8 HWC RGB -> uint8 LR RGB, MATLAB bicubic + ToPILImage truncation (matches the existing LR)."""
    return synthesize_lr([hr_rgb], scale)


def apply_setting(frames: np.ndarray, setting: dict, fps: float) -> tuple[np.ndarray, float]:
    if setting["codec"] == "none":
        return frames, 0.0
    if setting["codec"] == "chain":
        return chain_roundtrip(frames, [tuple(s) for s in setting["stages"]], fps=fps)
    return codec_roundtrip(frames, setting["codec"], setting["crf"], fps=fps)


def sample_setting(rng: np.random.Generator, codecs: list[str], tiers: dict, clean_prob: float) -> dict:
    if rng.random() < clean_prob:
        return {"codec": "none"}
    codec = codecs[rng.integers(len(codecs))]
    tier = ["high", "medium", "low"][rng.integers(3)]
    lo, hi = tiers[codec][tier]
    return {"codec": codec, "tier": tier, "crf": int(rng.integers(lo, hi + 1))}


def sample_platform(rng: np.random.Generator, args) -> dict:
    """Capture + share chain: phone-like HEVC at HR -> platform scaler -> platform encode(s) at LR."""
    stages = [(args.platform_codecs[rng.integers(len(args.platform_codecs))],
               int(rng.integers(args.platform_crf[0], args.platform_crf[1] + 1)))]
    if rng.random() < args.platform_reshare_prob:  # forwarded again: one more, slightly harsher, encode
        stages.append((stages[0][0], min(51, stages[0][1] + int(rng.integers(0, 4)))))
    return {
        "codec": "platform",
        "hr_codec": "x265", "hr_crf": int(rng.integers(args.platform_hr_crf[0], args.platform_hr_crf[1] + 1)),
        "hr_preset": "veryfast",  # hardware phone encoders trade efficiency for speed
        "kernel": args.platform_kernels[rng.integers(len(args.platform_kernels))],
        "stages": [list(st) for st in stages],
    }


def platform_lr(hr_frames, h: int, w: int, setting: dict, scale: int, fps: float):
    """Run the platform chain on an iterable of HR uint8 RGB frames.

    Returns ``(degraded LR [T, h/s, w/s, 3], clean MATLAB LR of the same HR frames, kb/s of the last stage)``.
    The HR-stage stream is round-tripped frame by frame (``stream_roundtrip``), so HR is never held whole.
    """
    clean = []

    def feed():
        for f in hr_frames:
            clean.append(downscale_to_uint8(to_float_chw(f), scale))
            yield f

    lr, _ = stream_roundtrip(
        feed(), h, w, setting["hr_codec"], setting["hr_crf"],
        on_frame=lambda f: downscale_kernel_to_uint8(to_float_chw(f), scale, setting["kernel"]),
        preset=setting.get("hr_preset"), fps=fps,
    )
    degraded, kbps = chain_roundtrip(np.stack(lr), [tuple(st) for st in setting["stages"]], fps=fps)
    return degraded, np.stack(clean), kbps


def sample_exposure(rng: np.random.Generator, exposures: list[int], weights: list[float] | None) -> int:
    if len(exposures) == 1:
        return exposures[0]
    p = None if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    return int(rng.choice(exposures, p=p))


def list_clips(env: lmdb.Environment) -> dict[str, list[str]]:
    """clip -> keys sorted by frame number, straight from the LMDB (no key-template guessing)."""
    clips = defaultdict(list)
    with env.begin() as txn:
        for key in txn.cursor().iternext(values=False):
            key = key.decode("ascii")
            clip, frame = key.split("/")
            clips[clip].append(key)
    for clip, keys in clips.items():
        keys.sort(key=lambda k: int(k.split("/")[1]))
        idx = [int(k.split("/")[1]) for k in keys]
        if idx != list(range(idx[0], idx[0] + len(idx))):
            raise ValueError(f"Clip {clip} has non-contiguous frame numbers; phase streams would be wrong")
    return dict(clips)


def meta_info_clips(meta_info: str) -> list[str]:
    with open(meta_info) as f:
        return [line.split(" ")[0] for line in f if line.strip()]


# ----------------------------------------------------------------------------- train
_W = {}


def _init_worker(hr_lmdb: str, scale: int, fps: float, tsf: int, exposure_space: str, lmdb_format: str):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _W["env"] = lmdb.open(hr_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    _W.update(scale=scale, fps=fps, tsf=tsf, space=exposure_space, fmt=lmdb_format)


def _hr_rgb(txn, key: str) -> np.ndarray:
    return unpack_raw(txn.get(key.encode("ascii")))[:, :, ::-1]


def _encode_stream(task: dict) -> dict:
    """One (clip, phase) stream: read HR, integrate exposure, downscale, round trip, pack. Runs in a worker."""
    keys, clip_keys, setting, n_exp = task["keys"], task["clip_keys"], task["setting"], task["exposure"]
    first = int(clip_keys[0].split("/")[1])  # clip frames are contiguous (list_clips), so position = idx - first
    blur_psnr = []
    if setting["codec"] == "platform":
        with _W["env"].begin() as txn:
            h, w = _hr_rgb(txn, keys[0]).shape[:2]

            def hr_frames():
                for k in keys:
                    window = exposure_window(int(k.split("/")[1]) - first, n_exp, 0, len(clip_keys) - 1)
                    x = integrate_exposure([_hr_rgb(txn, clip_keys[j]) for j in window], _W["space"])
                    yield x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()

            degraded, clean, kbps = platform_lr(hr_frames(), h, w, setting, _W["scale"], _W["fps"])
        return _finish_stream(task, degraded, clean, kbps, blur_psnr)
    with _W["env"].begin() as txn:
        lrs = []
        for k in keys:
            window = exposure_window(int(k.split("/")[1]) - first, n_exp, 0, len(clip_keys) - 1)
            frames = [_hr_rgb(txn, clip_keys[j]) for j in window]
            lr = synthesize_lr(frames, _W["scale"], _W["space"])
            if n_exp > 1:  # blur strength alone, vs the sharp LR of the centre frame
                blur_psnr.append(psnr_uint8(lr, synthesize_lr([frames[n_exp // 2]], _W["scale"])))
            lrs.append(lr)
        clean = np.stack(lrs)
    degraded, kbps = apply_setting(clean, setting, _W["fps"])
    return _finish_stream(task, degraded, clean, kbps, blur_psnr)


def _finish_stream(task: dict, degraded: np.ndarray, clean: np.ndarray, kbps: float, blur_psnr: list) -> dict:
    keys, setting, n_exp = task["keys"], task["setting"], task["exposure"]
    per_frame = [psnr_uint8(d, c) for d, c in zip(degraded, clean)]
    values = [pack_value(np.ascontiguousarray(f[:, :, ::-1]), _W["fmt"]) for f in degraded]  # RGB -> BGR
    record = {
        "clip": task["clip"], "phase": task["phase"], "n_frames": len(keys), **setting,
        "exposure": n_exp, "shutter_deg": round(360.0 * n_exp / _W["tsf"], 1),
        "kbps": round(kbps, 1), "psnr": round(float(np.mean(np.minimum(per_frame, 99.0))), 3),
        "psnr_min": round(float(min(per_frame)), 3),
    }
    if blur_psnr:
        record["blur_psnr"] = round(float(np.mean(np.minimum(blur_psnr, 99.0))), 3)
    return {"keys": keys, "values": values, "record": record, "shape": clean.shape[1:]}


def check_reference(hr_env, ref_lmdb: str, clips: dict, scale: int, n: int = 4) -> None:
    """Our clean downscale vs the existing LR LMDB on a few frames: should be (near) identical."""
    ref = lmdb.open(ref_lmdb, readonly=True, lock=False, readahead=False)
    keys = [ks[len(ks) // 2] for ks in list(clips.values())[:n]]
    with hr_env.begin() as th, ref.begin() as tr:
        for k in keys:
            ours = downscale_rgb(unpack_raw(th.get(k.encode()))[:, :, ::-1], scale)
            v = tr.get(k.encode())
            if v is None:
                raise SystemExit(f"Reference LR LMDB has no key {k}: key sets differ, fix before building")
            theirs = unpack_raw(v)[:, :, ::-1]
            diff = np.abs(ours.astype(int) - theirs.astype(int))
            print(f"  ref check {k}: PSNR {psnr_uint8(ours, theirs):.2f} dB, max|diff| {diff.max()}")
            if diff.max() > 1:
                raise SystemExit("Clean downscale does not reproduce the reference LR (max diff > 1): kernel mismatch")
    ref.close()


def build_train(args) -> None:
    if osp.exists(args.output_lmdb):
        raise SystemExit(f"{args.output_lmdb} already exists; refusing to overwrite")
    tiers = DEFAULT_TIERS if args.tiers_json is None else json.load(open(args.tiers_json))
    for c in args.codecs:
        if c not in tiers:
            raise SystemExit(f"No tier ranges for codec {c}")
    for n in args.exposures:
        if n < 1 or n % 2 == 0 or n > args.tsf:
            raise SystemExit(f"Exposure {n}: must be odd (centred on the keyframe) and <= tsf={args.tsf} (360 deg)")
    if args.exposure_weights is not None and len(args.exposure_weights) != len(args.exposures):
        raise SystemExit("--exposure-weights needs one weight per --exposures entry")

    hr_env = lmdb.open(args.hr_lmdb, readonly=True, lock=False, readahead=False)
    clips = list_clips(hr_env)
    if args.meta_info:
        wanted = meta_info_clips(args.meta_info)
        missing = [c for c in wanted if c not in clips]
        if missing:
            raise SystemExit(f"meta_info clips missing from HR LMDB: {missing[:5]}")
        clips = {c: clips[c] for c in wanted}
    if args.max_clips:
        clips = dict(list(clips.items())[: args.max_clips])
    n_frames = sum(len(k) for k in clips.values())
    print(f"{len(clips)} clips, {n_frames} frames, tsf={args.tsf} -> {len(clips) * args.tsf} streams @ {args.fps} fps")

    # Sanity checks before hours of work.
    first = next(iter(clips.values()))
    with hr_env.begin() as txn:
        probe = np.stack(
            [downscale_rgb(unpack_raw(txn.get(k.encode()))[:, :, ::-1], args.scale) for k in first[: 8 * args.tsf : args.tsf]]
        )
    print(f"LR frame size {probe.shape[1]}x{probe.shape[2]}; lossless x264 round trip {lossless_check(probe):.2f} dB")
    if args.reference_lr_lmdb:
        check_reference(hr_env, args.reference_lr_lmdb, clips, args.scale)

    tasks = []
    for clip, keys in clips.items():
        for p in range(args.tsf):
            stream = [k for k in keys if int(k.split("/")[1]) % args.tsf == p]
            if not stream:
                continue
            rng = np.random.default_rng([args.seed, zlib.crc32(clip.encode()), p])
            # Separate stream for the exposure, so enabling blur leaves the codec draws unchanged.
            rng_exp = np.random.default_rng([args.seed, zlib.crc32(clip.encode()), p, 1])
            rng_plat = np.random.default_rng([args.seed, zlib.crc32(clip.encode()), p, 2])
            setting = sample_setting(rng, args.codecs, tiers, args.clean_prob)
            if args.platform_prob > 0 and rng_plat.random() < args.platform_prob:
                setting = sample_platform(rng_plat, args)
            tasks.append(
                {"clip": clip, "phase": p, "keys": stream, "clip_keys": keys,
                 "setting": setting,
                 "exposure": sample_exposure(rng_exp, args.exposures, args.exposure_weights)}
            )
    hr_env.close()

    lr_h, lr_w = probe.shape[1:3]
    per_img = lr_h * lr_w * 3 + HEADER_BYTES
    env = lmdb.open(args.output_lmdb, map_size=int(per_img * n_frames * 1.05) + (1 << 30))
    meta = open(osp.join(args.output_lmdb, "meta_info.txt"), "w")
    params = open(osp.join(args.output_lmdb, "degradation_params.jsonl"), "w")
    written, t0 = set(), time.time()
    txn = env.begin(write=True)
    pending = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.hr_lmdb, args.scale, args.fps, args.tsf, args.exposure_space,
                                                  args.lmdb_format)) as pool:
        for i, res in enumerate(pool.imap_unordered(_encode_stream, tasks), 1):
            for key, value in zip(res["keys"], res["values"]):
                txn.put(key.encode("ascii"), value)
                meta.write(f"{key}.png ({lr_h},{lr_w},3) {0 if args.lmdb_format == 'raw' else 1}\n")
                written.add(key)
            pending += len(res["keys"])
            if pending * per_img > (512 << 20):
                txn.commit()
                txn = env.begin(write=True)
                pending = 0
            r = res["record"]
            params.write(json.dumps(r) + "\n")
            if r["codec"] != "none" and r["psnr_min"] < r["psnr"] - 6:
                print(f"  WARNING {r['clip']} phase {r['phase']}: min frame PSNR {r['psnr_min']} vs mean {r['psnr']}"
                      " (dropped/duplicated frame?)")
            if i % 20 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} streams, {el / 60:.1f} min, ETA {el / i * (len(tasks) - i) / 60:.1f} min",
                      flush=True)
    txn.commit()
    env.close()
    meta.close()
    params.close()

    expected = {k for ks in clips.values() for k in ks}
    if written != expected:
        raise SystemExit(f"Key set mismatch: {len(expected - written)} missing, {len(written - expected)} extra")
    summarize(osp.join(args.output_lmdb, "degradation_params.jsonl"))
    print(f"Done: {len(written)} frames -> {args.output_lmdb}")


def summarize(params_path: str) -> None:
    records = [json.loads(line) for line in open(params_path)]
    groups = defaultdict(list)
    for r in records:
        groups[(r["codec"], r.get("tier", "-"))].append(r)
    print(f"\n{'codec':6s} {'tier':7s} {'streams':>7s} {'kb/s':>8s} {'PSNR':>7s}")
    for (codec, tier), rs in sorted(groups.items()):
        print(f"{codec:6s} {tier:7s} {len(rs):7d} {np.mean([r['kbps'] for r in rs]):8.1f} "
              f"{np.mean([min(r['psnr'], 99) for r in rs]):7.2f}")
    by_exp = defaultdict(list)
    for r in records:
        by_exp[r.get("exposure", 1)].append(r)
    if len(by_exp) > 1 or 1 not in by_exp:
        print(f"\n{'exposure':>8s} {'shutter':>8s} {'streams':>7s} {'blur PSNR':>9s}")
        for n, rs in sorted(by_exp.items()):
            bp = [r["blur_psnr"] for r in rs if "blur_psnr" in r]
            print(f"{n:8d} {rs[0]['shutter_deg']:7.0f}d {len(rs):7d} {np.mean(bp) if bp else float('inf'):9.2f}")


# ----------------------------------------------------------------------------- val
def parse_setting(spec: str) -> dict:
    """``clean`` | ``[blur<n>+][hr<codec>:<crf>+k<kernel>+]codec:crf[+codec:crf...]`` -> setting dict.

    ``hr<codec>:<crf>`` (e.g. ``hrx265:20``) encodes at HR before the downscale and ``k<kernel>``
    (e.g. ``kbilinear``) picks the downscale kernel: together the platform chain, e.g. a
    WhatsApp-like ``hrx265:20+kbilinear+x264:32`` (phone HEVC -> platform scaler -> H.264).
    """
    if spec == "clean":
        return {"codec": "none", "exposure": 1}
    parts = spec.split("+")
    exposure = 1
    if parts[0].startswith("blur"):
        exposure = int(parts.pop(0)[4:])
    hr_stage, kernel = None, None
    if parts and parts[0].startswith("hr"):
        c, q = parts.pop(0)[2:].split(":")
        hr_stage = (c, int(q))
    if parts and parts[0].startswith("k") and parts[0][1:] in DOWNSCALE_KERNELS:
        kernel = parts.pop(0)[1:]
    if hr_stage or kernel:
        stages = [(st.split(":")[0], int(st.split(":")[1])) for st in parts]
        if hr_stage is None or not stages:
            raise SystemExit(f"Platform setting '{spec}' needs hr<codec>:<crf>, then optional k<kernel>, then LR stages")
        return {"codec": "platform", "hr_codec": hr_stage[0], "hr_crf": hr_stage[1], "hr_preset": "veryfast",
                "kernel": kernel or "bicubic", "stages": [list(st) for st in stages], "exposure": exposure}
    stages = [(s.split(":")[0], int(s.split(":")[1])) for s in parts]
    if not stages:
        return {"codec": "none", "exposure": exposure}
    if len(stages) == 1:
        return {"codec": stages[0][0], "crf": stages[0][1], "exposure": exposure}
    return {"codec": "chain", "stages": stages, "exposure": exposure}


def setting_tag(spec: str) -> str:
    return "clean" if spec == "clean" else spec.replace(":", "crf").replace("+", "_then_")


def _frame_number(name: str):
    """Last number in the file stem, as ``dataset_video_test._extract_frame_number``."""
    digits = re.findall(r"\d+", osp.splitext(osp.basename(name))[0])
    return int(digits[-1]) if digits else None


def _read_list(list_dir: str, clip: str, hr_names: list[str]) -> list[str]:
    """Keyframe names from ``<clip>_im_list.txt`` (VideoRecurrentTestDataset convention), matched by frame number."""
    by_num = {_frame_number(n): n for n in hr_names}
    path = osp.join(list_dir, f"{clip}_im_list.txt")
    if not osp.isfile(path):
        raise SystemExit(f"No keyframe list for {clip}: {path}")
    names = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        num = _frame_number(line)
        if num not in by_num:
            raise SystemExit(f"{clip}: listed frame '{line}' not found among HR frames")
        names.append(by_num[num])
    return names


def build_val(args) -> None:
    clips = sorted(d for d in os.listdir(args.hr_dir) if osp.isdir(osp.join(args.hr_dir, d)))
    if args.max_clips:
        clips = clips[: args.max_clips]
    gt_root = None
    if args.gt_stride:
        gt_root = osp.join(args.output_root, f"hr_stride{args.gt_stride}")
    outs = [osp.join(args.output_root, f"{args.prefix}_{setting_tag(spec)}") for spec in args.settings]
    for out in outs + ([gt_root] if gt_root else []):
        if osp.exists(out):
            raise SystemExit(f"{out} already exists; refusing to overwrite")
    for clip in clips:
        hr_names = sorted(f for f in os.listdir(osp.join(args.hr_dir, clip)) if f.lower().endswith(".png"))
        names = _read_list(args.list_dir, clip, hr_names)
        if gt_root:
            link_sparse_gt(args.hr_dir, gt_root, clip, hr_names, names, args.gt_stride)
        if not args.settings:
            continue
        # Exposure windows are read from the dense HR folder around each keyframe.
        num_to_pos = {_frame_number(n): i for i, n in enumerate(hr_names)}
        cache, lr_by_exposure = {}, {}

        def hr_frame(i):
            if i not in cache:
                cache[i] = np.asarray(Image.open(osp.join(args.hr_dir, clip, hr_names[i])).convert("RGB"))
            return cache[i]

        for n_exp in sorted({parse_setting(spec)["exposure"] for spec in args.settings}):
            lr_by_exposure[n_exp] = np.stack([
                synthesize_lr([hr_frame(j) for j in exposure_window(num_to_pos[_frame_number(k)], n_exp, 0,
                                                                       len(hr_names) - 1)], args.scale,
                              args.exposure_space)
                for k in names
            ])
        clean = lr_by_exposure.get(1)
        line = f"{clip}: {len(names)} keyframes"
        for spec in args.settings:
            setting = parse_setting(spec)
            if setting["codec"] == "platform":
                n_exp = setting["exposure"]
                hr_iter = (
                    integrate_exposure([hr_frame(j) for j in exposure_window(num_to_pos[_frame_number(k)], n_exp, 0,
                                                                             len(hr_names) - 1)],
                                       args.exposure_space).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
                    for k in names
                )
                h0, w0 = hr_frame(num_to_pos[_frame_number(names[0])]).shape[:2]
                degraded, _, kbps = platform_lr(hr_iter, h0, w0, setting, args.scale, args.fps)
            else:
                degraded, kbps = apply_setting(lr_by_exposure[setting["exposure"]], setting, args.fps)
            out_dir = osp.join(args.output_root, f"{args.prefix}_{setting_tag(spec)}", clip)
            os.makedirs(out_dir)
            for frame, name in zip(degraded, names):
                Image.fromarray(frame).save(osp.join(out_dir, name), compress_level=1)
            if spec != "clean" and clean is not None:
                line += f" | {spec} {psnr_uint8(degraded, clean):.2f} dB {kbps:.0f} kb/s"
        cache.clear()
        print(line, flush=True)
    if args.settings:
        print(f"Done. Point the test dataroot_lq at {args.output_root}/{args.prefix}_<setting> (same test_list_dir).")
    if gt_root:
        print(f"Point the test dataroot_gt at {gt_root} (tsf = keyframe spacing / {args.gt_stride}).")


def link_sparse_gt(hr_dir: str, gt_root: str, clip: str, hr_names: list[str], keyframes: list[str], stride: int):
    """Symlink every ``stride``-th HR frame (phase-locked to the first keyframe) into ``gt_root/<clip>``.

    Validation with ``test_list_dir`` compares the model output frame-by-frame against the GT folder in order
    (``main_train_elvsr.py`` ``_run_validation``), with no temporal stride. At an output rate below the source
    rate (e.g. tsf=2 on a 30 fps input from 240 fps: 60 fps out, stride 4), the GT folder must therefore hold
    exactly the output timestamps, or every intermediate frame is scored against the wrong GT frame.
    """
    first, last = _frame_number(keyframes[0]), _frame_number(keyframes[-1])
    if any((_frame_number(k) - first) % stride for k in keyframes):
        raise SystemExit(f"{clip}: keyframe spacing is not a multiple of --gt-stride {stride}")
    out_dir = osp.join(gt_root, clip)
    os.makedirs(out_dir)
    for name in hr_names:
        n = _frame_number(name)
        if first <= n <= last and (n - first) % stride == 0:
            os.symlink(osp.abspath(osp.join(hr_dir, clip, name)), osp.join(out_dir, name))


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="HR raw LMDB -> dense compressed LR raw LMDB")
    tr.add_argument("--hr-lmdb", required=True)
    tr.add_argument("--output-lmdb", required=True)
    tr.add_argument("--meta-info", default=None, help="Training meta_info file; restricts to its clips")
    tr.add_argument("--reference-lr-lmdb", default=None, help="Existing clean LR LMDB to verify the downscale against")
    tr.add_argument("--codecs", nargs="+", default=["x264", "x265"], choices=sorted(DEFAULT_TIERS))
    tr.add_argument("--clean-prob", type=float, default=0.15)
    tr.add_argument("--tiers-json", default=None, help="JSON {codec: {tier: [crf_lo, crf_hi]}} overriding the defaults")
    tr.add_argument("--workers", type=int, default=16)
    tr.add_argument("--lmdb-format", choices=("raw", "png"), default="raw",
                    help="Output value format: raw (loader raw_lmdb: true) or png (raw_lmdb: false). The HR LMDB "
                         "is read in either format. Must match the GT LMDB, since raw_lmdb applies to both.")
    tr.add_argument("--max-clips", type=int, default=None, help="Smoke-test on the first N clips")
    tr.add_argument("--platform-prob", type=float, default=0.0,
                    help="Fraction of streams drawn from the capture+share chain instead of a single LR encode "
                         "(default 0 = off): phone HEVC at HR -> platform downscale kernel -> platform encode(s)")
    tr.add_argument("--platform-hr-crf", type=int, nargs=2, default=[18, 24], help="Phone-stage x265 CRF range at HR")
    tr.add_argument("--platform-kernels", nargs="+", default=["bicubic", "bilinear", "area", "lanczos"],
                    choices=DOWNSCALE_KERNELS, help="Platform scaler kernels, sampled uniformly")
    tr.add_argument("--platform-codecs", nargs="+", default=["x264"], choices=sorted(CODECS),
                    help="Platform-stage codecs at LR (WhatsApp: x264; YouTube: x264/vp9)")
    tr.add_argument("--platform-crf", type=int, nargs=2, default=[28, 36], help="Platform-stage CRF range at LR")
    tr.add_argument("--platform-reshare-prob", type=float, default=0.3,
                    help="Probability of a second platform encode (a forwarded video)")
    tr.add_argument("--exposures", type=int, nargs="+", default=[1],
                    help="Exposure lengths in source frames (odd, <= tsf), sampled per stream; default 1 = no blur. "
                         "E.g. 1 3 5 7 = 45/135/225/315 deg shutter at 30 fps from 240 fps")
    tr.add_argument("--exposure-weights", type=float, nargs="+", default=None,
                    help="Sampling weights for --exposures (default uniform)")

    va = sub.add_parser("val", help="HR frame folders + keyframe lists -> one LR folder per setting")
    va.add_argument("--hr-dir", required=True, help="<hr-dir>/<clip>/<frame>.png")
    va.add_argument("--list-dir", required=True, help="Folder of <clip>_im_list.txt keyframe lists (test_list_dir)")
    va.add_argument("--output-root", required=True)
    va.add_argument("--prefix", default="lr")
    va.add_argument("--gt-stride", type=int, default=None,
                    help="Also write hr_stride<N>/ (symlinks to every N-th HR frame): the val GT for tsf = spacing / N")
    va.add_argument("--settings", nargs="*", default=DEFAULT_VAL_SETTINGS,
                    help="'clean' or codec:crf[+codec:crf...], codecs x264/x265/av1; pass none to only write --gt-stride GT")
    va.add_argument("--max-clips", type=int, default=None)

    for p in (tr, va):
        p.add_argument("--exposure-space", choices=("linear", "srgb"), default="linear",
                       help="Average exposure frames in linear light (physical) or in sRGB (GoPro-dataset style)")
        p.add_argument("--scale", type=int, default=4)
        p.add_argument("--fps", type=float, default=30.0, help="Frame rate of the encoded (sparse) stream")
    tr.add_argument("--tsf", type=int, default=8)
    tr.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    build_train(args) if args.cmd == "train" else build_val(args)


if __name__ == "__main__":
    main()
