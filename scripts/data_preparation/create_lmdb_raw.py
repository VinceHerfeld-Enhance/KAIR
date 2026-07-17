"""Build a *raw-uint8* LMDB from a folder of PNG frames.

Unlike ``create_lmdb.py`` (which stores PNG-encoded bytes), this stores each
frame as **uncompressed HWC BGR uint8** with a tiny self-describing header:

    [ 4B magic b"RAW1" ][ 12B: h,w,c as 3x little-endian uint32 ][ h*w*c bytes raw HWC BGR uint8 ]

This trades ~10x storage for zero decode at train time (the dataloader just
reshapes) and enables spatial crop-before-decode via
``LmdbBackend.get_raw_crop`` (see ``utils/utils_video.py``). Pixels are
bit-identical to the PNGs (PNG is lossless), so training numerics are unchanged.

Usage (run from KAIR/):
    python scripts/data_preparation/create_lmdb_raw.py --dataset REDS
    # or explicit folders:
    python scripts/data_preparation/create_lmdb_raw.py \
        --input-folder /path/to/train_sharp --output-lmdb /path/to/train_sharp_raw.lmdb

Edit the path constants in ``create_lmdb_for_reds`` for your machine, or pass
``--input-folder``/``--output-lmdb`` to override.
"""

import argparse
import sys
from multiprocessing import Pool
from os import path as osp

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

kair_root = osp.abspath(osp.join(osp.dirname(__file__), "../.."))
sys.path.append(kair_root)

from utils.utils_video import scandir, RAW_LMDB_MAGIC  # noqa: E402

HEADER_BYTES = 16  # 4 (magic) + 3 * 4 (h, w, c uint32)


def pack_raw(img):
    """Pack an HWC BGR uint8 image into a raw-uint8 LMDB value."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, c = img.shape
    header = RAW_LMDB_MAGIC + np.array([h, w, c], dtype="<u4").tobytes()
    return header + img.tobytes(), (h, w, c)


def read_raw_img_worker(path, key):
    """Decode a PNG and return its raw-uint8 LMDB value.

    Returns:
        tuple: (key, packed_bytes, (h, w, c)).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # HWC BGR uint8, always 3 channels
    # deal with `libpng error: Read Error` (mirror utils_lmdb.read_img_worker)
    if img is None:
        print(f"To deal with `libpng error: Read Error`, use PIL to load {path}")
        from PIL import Image

        pil = Image.open(path).convert("RGB")
        img = np.asanyarray(pil)[:, :, [2, 1, 0]]  # RGB -> BGR
    value, shape = pack_raw(img)
    return key, value, shape


def read_raw_img_worker_star(args):
    """Helper to unpack worker arguments for imap_unordered."""
    return read_raw_img_worker(*args)


def make_raw_lmdb_from_imgs(
    data_path,
    lmdb_path,
    img_path_list,
    keys,
    batch=None,
    n_thread=40,
    multiprocessing_chunksize=8,
    map_size=None,
):
    """Make a raw-uint8 lmdb from images (parallel stream mode, low peak RAM).

    Args:
        data_path (str): Root folder for the images.
        lmdb_path (str): Output ``*.lmdb`` path.
        img_path_list (list[str]): Image paths relative to ``data_path``.
        keys (list[str]): LMDB keys (e.g. ``000/00000000``).
        batch (int | None): Commit every ``batch`` writes. If None, auto-tuned to
            ~512 MiB of buffered values (raw HR frames are large, so a fixed count
            would buffer many GiB in the write txn).
        n_thread (int): Decode worker processes. Default: 40.
        multiprocessing_chunksize (int): imap chunksize. Default: 8.
        map_size (int | None): LMDB map size in bytes. If None, estimated from
            the first image (raw size is deterministic).
    """
    assert len(img_path_list) == len(keys), (
        f"img_path_list and keys length mismatch: {len(img_path_list)} vs {len(keys)}"
    )
    if not lmdb_path.endswith(".lmdb"):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f"Folder {lmdb_path} already exists. Exit.")
        sys.exit(1)

    print(f"Create RAW lmdb for {data_path}, save to {lmdb_path}...")
    print(f"Total images: {len(img_path_list)}")

    # Estimate map size from the first image (all raw values of a given shape are
    # exactly h*w*c + HEADER_BYTES; add 5% + 1 GiB for lmdb page/B-tree overhead).
    if map_size is None:
        probe = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_COLOR)
        if probe is None:
            raise RuntimeError(f"Could not read probe image {img_path_list[0]}")
        h, w, c = probe.shape
        per_img = h * w * c + HEADER_BYTES
        print(f"Raw size per image: {per_img} bytes ({h}x{w}x{c})")
        map_size = int(per_img * len(img_path_list) * 1.05) + (1 << 30)
    else:
        probe = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_COLOR)
        per_img = probe.shape[0] * probe.shape[1] * probe.shape[2] + HEADER_BYTES
    print(f"LMDB map_size: {map_size / (1 << 30):.1f} GiB")

    if batch is None:
        # Target ~512 MiB of buffered values per write txn to cap peak RAM.
        batch = max(1, int((512 << 20) / per_img))
    print(f"Commit batch: {batch} images (~{batch * per_img / (1 << 20):.0f} MiB/txn)")

    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, "meta_info.txt"), "w")

    print(f"Read + write with multiprocessing, #thread: {n_thread}, chunksize: {multiprocessing_chunksize} ...")
    pbar = tqdm(total=len(img_path_list), unit="image")
    pool = Pool(n_thread)
    worker_args = ((osp.join(data_path, path), key) for path, key in zip(img_path_list, keys))
    for idx, (key, value, shape) in enumerate(
        pool.imap_unordered(read_raw_img_worker_star, worker_args, chunksize=multiprocessing_chunksize)
    ):
        pbar.update(1)
        pbar.set_description(f"Write {key}")
        h, w, c = shape
        txn.put(key.encode("ascii"), value)
        # meta_info kept in the create_lmdb.py textual format (loader ignores it).
        txt_file.write(f"{key}.png ({h},{w},{c}) 0\n")
        if (idx + 1) % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pool.close()
    pool.join()
    pbar.close()

    txn.commit()
    env.close()
    txt_file.close()
    print("\nFinish writing raw lmdb.")


def prepare_keys(folder_path):
    """List PNG frames (recursively) and build lmdb keys.

    Keys are the frame path without extension, e.g. ``000/00000000`` (REDS) or
    ``clip/00000000`` (Adobe240). Dataset-agnostic: the layout is identical.
    """
    print("Reading image path list ...")
    img_path_list = sorted(list(scandir(folder_path, suffix="png", recursive=True)))
    keys = [v.split(".png")[0] for v in img_path_list]
    return img_path_list, keys


def _build(input_folder, output_lmdb, **kwargs):
    img_path_list, keys = prepare_keys(input_folder)
    make_raw_lmdb_from_imgs(input_folder, output_lmdb, img_path_list, keys, **kwargs)


def create_lmdb_for_reds(input_folder=None, output_lmdb=None):
    """Create raw lmdb files for the REDS dataset (GT + LQ). Edit the constants."""
    if input_folder and output_lmdb:
        _build(input_folder, output_lmdb)
        return
    root = "/home/vherfeld/storage/vherfeld/datasets/REDS"
    # GT (train_sharp) — where the decode win is.
    _build(f"{root}/train_sharp", f"{root}/train_sharp_with_val_raw.lmdb")
    # LQ (train_sharp_bicubic/X4) — optional (LQ decode is ~16x cheaper).
    _build(f"{root}/train_sharp_bicubic/X4", f"{root}/train_sharp_bicubic_with_val_raw.lmdb")


def create_lmdb_for_adobe240(input_folder=None, output_lmdb=None):
    """Create raw lmdb files for the Adobe240 dataset (HR + LR). Edit the constants.

    Mirrors create_lmdb.create_lmdb_for_adobe240 (clip/00000000 keys, stream mode).
    Point the training config's dataroot_gt/lq at these ``*_raw.lmdb`` outputs and
    set ``raw_lmdb: true``.
    """
    if input_folder and output_lmdb:
        _build(input_folder, output_lmdb, n_thread=12, multiprocessing_chunksize=16)
        return
    root = "/home/vherfeld/storage/vherfeld/datasets/Adobe240"
    # HR — the decode win.
    _build(f"{root}/hr", f"{root}/train_hr_raw.lmdb", n_thread=12, multiprocessing_chunksize=16)
    # LR — optional.
    _build(f"{root}/lr", f"{root}/train_lr_raw.lmdb", n_thread=12, multiprocessing_chunksize=16)


_DATASETS = {"reds": create_lmdb_for_reds, "adobe240": create_lmdb_for_adobe240}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a raw-uint8 LMDB from PNG frames.")
    parser.add_argument("--dataset", type=str, default="REDS", help="Preset. Options: 'REDS', 'Adobe240'.")
    parser.add_argument(
        "--input-folder", type=str, default=None, help="Override: folder of PNG frames to encode (recursive)."
    )
    parser.add_argument(
        "--output-lmdb", type=str, default=None, help="Override: output *.lmdb path (used with --input-folder)."
    )
    args = parser.parse_args()

    ds = args.dataset.lower()
    if ds not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Options: {sorted(_DATASETS)}")
    if args.input_folder is not None or args.output_lmdb is not None:
        if not (args.input_folder and args.output_lmdb):
            parser.error("--input-folder and --output-lmdb must be given together.")
        _DATASETS[ds](args.input_folder, args.output_lmdb)
    else:
        _DATASETS[ds]()
