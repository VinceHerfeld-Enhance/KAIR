"""Partition a raw Adobe240 frame tree into disjoint train / test roots.

Adobe240 ships no official C-STVSR split. The de-facto one originates with
VideoINR and is publicly available only through BF-STVSR's
``data/adobe240fps_folder_{train,test}.txt`` (100 train / 17 test clips, disjoint;
the remaining videos are used by neither). This script materialises that split on
disk so the lmdb build can simply point at the clean train root.

Expected input layout (as consumed by ``create_lmdb.py``)::

    <root>/hr/<clip>/<frame>.png
    <root>/lr/<clip>/<frame>.png

Output layout::

    <out-root>/Adobe240_train/hr/<clip>/...
    <out-root>/Adobe240_train/lr/<clip>/...
    <out-root>/Adobe240_test/hr/<clip>/...
    <out-root>/Adobe240_test/lr/<clip>/...

By default clips are **symlinked**, which costs no disk and no copy time.
``utils.utils_videoio.scandir`` (used by ``create_lmdb.py``) resolves symlinks,
so the lmdb build sees the frames normally. Use ``--mode copy`` if the split must
survive independently of the source tree, or ``--mode move`` to relocate.

Typical use, matching BF-STVSR exactly::

    python split_adobe240.py \
        --root      /path/to/Adobe240 \
        --out-root  /path/to/Adobe240_split \
        --train-list /path/to/BF-STVSR/data/adobe240fps_folder_train.txt

Then build the lmdb from ``<out-root>/Adobe240_train/{hr,lr}``.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# The standard 17-sequence Adobe240 C-STVSR test set, matching BF-STVSR
# ``data/adobe240fps_folder_test.txt`` (the VideoINR/MoTIF/BF-STVSR lineage).
ADOBE240_TEST_VIDEOS = [
    "GOPR9653",
    "IMG_0001",
    "IMG_0003",
    "IMG_0004a",
    "IMG_0004b",
    "IMG_0012",
    "IMG_0013",
    "IMG_0015",
    "IMG_0016",
    "IMG_0017",
    "IMG_0024",
    "IMG_0037",
    "IMG_0037a",
    "IMG_0045",
    "IMG_0056",
    "IMG_0058",
    "IMG_0175",
]


def read_name_list(path: Path) -> list[str]:
    """Read a newline-separated clip-name list, tolerating a missing final newline."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def place(src: Path, dst: Path, mode: str) -> None:
    """Materialise ``src`` at ``dst`` using the requested strategy."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"unknown mode {mode!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Adobe240 root containing the --subdirs trees.")
    parser.add_argument("--out-root", type=Path, required=True, help="Where Adobe240_train / Adobe240_test are created.")
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["hr", "lr"],
        help="Parallel frame trees under --root to partition. Default: hr lr",
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=None,
        help="Clip names for the train split (e.g. BF-STVSR adobe240fps_folder_train.txt). "
        "If omitted, every clip that is not in the test list goes to train.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Clip names for the test split. Defaults to the built-in 17-clip Adobe240 test set.",
    )
    parser.add_argument("--train-name", default="Adobe240_train", help="Output dir name for the train split.")
    parser.add_argument("--test-name", default="Adobe240_test", help="Output dir name for the test split.")
    parser.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink", help="Default: symlink.")
    parser.add_argument("--dry-run", action="store_true", help="Report the plan without touching the filesystem.")
    parser.add_argument("--force", action="store_true", help="Proceed despite missing clips or non-empty outputs.")
    args = parser.parse_args()

    # Writing the split inside the source tree makes the two trees hard to tell apart
    # later, and a rerun would then rescan its own output. Refuse it.
    root_res, out_res = args.root.resolve(), args.out_root.resolve()
    if out_res == root_res or root_res in out_res.parents:
        print(f"ERROR: --out-root ({out_res}) must not be inside --root ({root_res}).", file=sys.stderr)
        raise SystemExit(1)

    test_names = read_name_list(args.test_list) if args.test_list else list(ADOBE240_TEST_VIDEOS)
    train_names = read_name_list(args.train_list) if args.train_list else None

    overlap = sorted(set(test_names) & set(train_names)) if train_names else []
    if overlap:
        print(f"ERROR: train and test lists overlap on {len(overlap)} clip(s): {', '.join(overlap)}", file=sys.stderr)
        raise SystemExit(1)

    # Clip inventory is taken from the first subdir; the others must agree.
    probe = args.root / args.subdirs[0]
    if not probe.is_dir():
        print(f"ERROR: {probe} is not a directory.", file=sys.stderr)
        raise SystemExit(1)
    on_disk = sorted(d.name for d in probe.iterdir() if d.is_dir() and not d.name.startswith("."))

    if train_names is None:
        train_names = [c for c in on_disk if c not in set(test_names)]

    assigned = set(train_names) | set(test_names)
    unassigned = [c for c in on_disk if c not in assigned]
    missing_train = sorted(set(train_names) - set(on_disk))
    missing_test = sorted(set(test_names) - set(on_disk))

    print(f"root                 : {args.root}")
    print(f"subdirs              : {', '.join(args.subdirs)}")
    print(f"mode                 : {args.mode}")
    print(f"clips on disk        : {len(on_disk)}")
    print(f"train clips          : {len(train_names)}")
    print(f"test  clips          : {len(test_names)}")
    print(f"unassigned (neither) : {len(unassigned)}")
    if unassigned:
        print(f"  -> {', '.join(unassigned)}")
        print("     (BF-STVSR's split also leaves videos in neither list; these are simply unused.)")

    if missing_train or missing_test:
        print("", file=sys.stderr)
        if missing_train:
            print(f"WARNING: {len(missing_train)} train clip(s) not found on disk: {', '.join(missing_train)}", file=sys.stderr)
        if missing_test:
            print(f"WARNING: {len(missing_test)} test clip(s) not found on disk: {', '.join(missing_test)}", file=sys.stderr)
        print(f"First 10 clip names on disk: {', '.join(on_disk[:10])}", file=sys.stderr)
        print("A naming mismatch here means the split is not doing what you think.", file=sys.stderr)
        if not args.force:
            print("Refusing to proceed. Re-run with --force once you have confirmed the naming.", file=sys.stderr)
            raise SystemExit(1)

    splits = {args.train_name: train_names, args.test_name: test_names}

    for split_name, clips in splits.items():
        for subdir in args.subdirs:
            out_dir = args.out_root / split_name / subdir
            if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
                print(f"\nERROR: {out_dir} exists and is not empty. Use --force to add to it.", file=sys.stderr)
                raise SystemExit(1)

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return

    total = 0
    for split_name, clips in splits.items():
        for subdir in args.subdirs:
            src_root = args.root / subdir
            out_dir = args.out_root / split_name / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            for clip in clips:
                src = src_root / clip
                if not src.is_dir():
                    continue
                place(src, out_dir / clip, args.mode)
                total += 1
        print(f"{split_name:16s}: {sum(1 for c in clips if (args.root / args.subdirs[0] / c).is_dir())} clips")

    print(f"\nplaced {total} clip directories under {args.out_root}")
    print("\nNext: point create_lmdb.py's folder_path at")
    for subdir in args.subdirs:
        print(f"  {args.out_root / args.train_name / subdir}")


if __name__ == "__main__":
    main()
