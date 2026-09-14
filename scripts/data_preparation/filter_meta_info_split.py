"""Filter a KAIR ``meta_info`` file down to a clip-level train split.

``VideoRecurrentTrainDataset`` builds ``self.keys`` exclusively from the
``meta_info_file`` (see ``KAIR/data/dataset_video_train.py``), and every read in
``__getitem__`` resolves its ``clip_name`` from that list. Clips absent from the
meta_info are therefore never enumerated, never sampled and never read -- even
when they are physically present in the lmdb. Filtering this file is a complete
way to remove evaluation clips from training without rebuilding the lmdb.

The default exclusion list is the 17-sequence Adobe240 C-STVSR test set used by
the VideoINR / MoTIF / BF-STVSR lineage (``BF-STVSR/data/adobe240fps_folder_test.txt``).

Usage (exclude the canonical Adobe240 test clips)::

    python filter_meta_info_split.py \
        --meta-info /path/to/meta_info_Adobe240_train_GT.txt \
        --out       /path/to/meta_info_Adobe240_train_GT.clean.txt

Usage (keep exactly the clips named in BF-STVSR's train list)::

    python filter_meta_info_split.py \
        --meta-info /path/to/meta_info_Adobe240_train_GT.txt \
        --out       /path/to/meta_info_Adobe240_train_GT.clean.txt \
        --keep-file /path/to/adobe240fps_folder_train.txt
"""

import argparse
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
    names = []
    for line in path.read_text().splitlines():
        name = line.strip()
        if name:
            names.append(name)
    return names


def clip_of(line: str) -> str:
    """meta_info line layout: ``<folder> <frame_num> (<h>,<w>,<c>) <start_frame>``."""
    return line.split(" ")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-info", type=Path, required=True, help="Input meta_info file.")
    parser.add_argument("--out", type=Path, required=True, help="Output (filtered) meta_info file.")
    parser.add_argument(
        "--keep-file",
        type=Path,
        default=None,
        help="Clip-name list to KEEP (e.g. BF-STVSR adobe240fps_folder_train.txt). Mutually exclusive with --exclude-file.",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Clip-name list to EXCLUDE. Defaults to the built-in Adobe240 17-clip test set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write the output even when the split does not match expectations (see the checks below).",
    )
    args = parser.parse_args()

    if args.keep_file is not None and args.exclude_file is not None:
        parser.error("--keep-file and --exclude-file are mutually exclusive.")

    lines = [ln for ln in args.meta_info.read_text().splitlines() if ln.strip()]
    clips_in_meta = []
    for ln in lines:
        name = clip_of(ln)
        if name not in clips_in_meta:
            clips_in_meta.append(name)

    if args.keep_file is not None:
        keep = set(read_name_list(args.keep_file))
        kept_lines = [ln for ln in lines if clip_of(ln) in keep]
        selected = sorted({clip_of(ln) for ln in kept_lines})
        missing = sorted(keep - set(selected))
        mode = f"keep-only from {args.keep_file}"
    else:
        exclude_path = args.exclude_file
        drop = set(read_name_list(exclude_path)) if exclude_path else set(ADOBE240_TEST_VIDEOS)
        kept_lines = [ln for ln in lines if clip_of(ln) not in drop]
        dropped = sorted({clip_of(ln) for ln in lines if clip_of(ln) in drop})
        missing = sorted(drop - set(dropped))
        selected = sorted({clip_of(ln) for ln in kept_lines})
        mode = f"exclude {'from ' + str(exclude_path) if exclude_path else 'built-in Adobe240 test set'}"

    print(f"mode                 : {mode}")
    print(f"meta_info            : {args.meta_info}")
    print(f"clips in input       : {len(clips_in_meta)}")
    print(f"clips in output      : {len(selected)}")
    print(f"lines  in input      : {len(lines)}")
    print(f"lines  in output     : {len(kept_lines)}")

    if args.keep_file is None:
        print(f"clips excluded       : {len(dropped)}")
        if dropped:
            print(f"  -> {', '.join(dropped)}")

    # A name mismatch is the dangerous failure: the filter silently does nothing and
    # training stays contaminated. Make it loud rather than letting it pass.
    if missing:
        print(f"\nWARNING: {len(missing)} name(s) from the list were NOT found in the meta_info:", file=sys.stderr)
        print(f"  {', '.join(missing)}", file=sys.stderr)
        print(
            "This usually means the clip naming in the meta_info differs from the list "
            "(e.g. numeric ids instead of IMG_xxxx). Check the names below before trusting the output.",
            file=sys.stderr,
        )
        print(f"\nFirst 10 clip names in the meta_info: {', '.join(clips_in_meta[:10])}", file=sys.stderr)
        if not args.force:
            print("\nRefusing to write. Re-run with --force once you have confirmed the naming.", file=sys.stderr)
            raise SystemExit(1)

    if not kept_lines:
        print("\nERROR: the filter kept zero lines; refusing to write.", file=sys.stderr)
        raise SystemExit(1)

    args.out.write_text("\n".join(kept_lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
