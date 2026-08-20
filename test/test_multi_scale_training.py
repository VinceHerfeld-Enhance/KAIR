"""Arbitrary-spatial-scale training: sampler contract and data geometry.

Run from ``KAIR/``::

    PYTHONPATH=.:../VSR/src ../VSR/.venv/bin/python -m pytest test/test_multi_scale_training.py -q

What is guarded here, and why each property is load-bearing:

* **One scale per optimizer step, identical on every rank.** Accelerate shards the
  *prepared loader*, so each process iterates this same batch sampler and keeps the
  batches at positions ``rank, rank+W, ...``. If the emitted sequence differed
  between ranks the per-rank slices would not partition the epoch — samples would
  be silently dropped and duplicated. Nothing downstream would raise.
* **GT crop == round(lq_patch * s).** The dataset infers the crop from the scale;
  if this drifts, the network's output and the GT stop matching and the loss is
  computed against a shifted target.
* **The LR lands on exactly ``lq_patch``.** ``imresize_gpu`` returns
  ``ceil(in*scale)``, so the resize must be driven by ``lq_patch / G`` rather than
  ``1 / s``. The latter is off by one for many scales.
* **The anchor frames are locatable inside a sparse GT tensor.** ``H`` holds only
  the input-aligned frames plus this sample's sampled mids, so the anchors are not
  at a fixed stride within it — ``lq_sel`` has to carry their positions.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.multi_scale_sampler import MultiScaleBatchSampler  # noqa: E402

LQ_PATCH = 64
TSF = 8
N_INTERP = 3
NUM_FRAME = 17  # (NUM_FRAME - 1) % TSF == 0  ->  T_in = 3
T_IN = (NUM_FRAME - 1) // TSF + 1
SCALE_RANGE = (1.2, 4.0)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def _sampler(**kw):
    defaults = dict(num_samples=100, batch_size=4, scale_range=SCALE_RANGE,
                    group_size=1, shuffle=True, drop_last=True, seed=0)
    defaults.update(kw)
    return MultiScaleBatchSampler(**defaults)


def test_items_are_index_scale_pairs_with_one_scale_per_batch():
    for batch in _sampler():
        assert len({scale for _, scale in batch}) == 1
        assert all(isinstance(i, int) for i, _ in batch)


def test_sequence_is_identical_across_ranks():
    """Required for correctness under Accelerate's positional batch sharding."""
    assert list(_sampler()) == list(_sampler())


def test_scale_is_shared_across_the_batches_of_one_step():
    """With ``group_size == world_size``, one optimizer step is single-scale."""
    world = 4
    batches = list(_sampler(group_size=world))
    for step in range(len(batches) // world):
        group = batches[step * world : (step + 1) * world]
        assert len({b[0][1] for b in group}) == 1


def test_epoch_changes_order_and_scales():
    sampler = _sampler()
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_scale_is_reproducible_from_seed_epoch_step():
    a, b = _sampler(), _sampler()
    b.set_epoch(0)
    assert [a.scale_for_step(k) for k in range(20)] == [b.scale_for_step(k) for k in range(20)]
    a.set_epoch(3)
    assert a.scale_for_step(0) != b.scale_for_step(0)


def test_no_duplicate_indices_within_an_epoch():
    batches = list(_sampler())
    flat = [i for b in batches for i, _ in b]
    assert len(flat) == len(set(flat))


def test_scales_stay_inside_the_requested_range():
    scales = [s for b in _sampler() for _, s in b]
    assert min(scales) >= SCALE_RANGE[0]
    assert max(scales) <= SCALE_RANGE[1]


def test_rejects_a_degenerate_range():
    with pytest.raises(ValueError):
        _sampler(scale_range=(4.0, 1.2))


# ---------------------------------------------------------------------------
# Dataset geometry (needs a raw-uint8 LMDB, built synthetically)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_gt_lmdb(tmp_path_factory):
    """A minimal raw-uint8 GT LMDB, same layout as create_lmdb_raw.py writes:
    ``[4B b"RAW1"][12B h,w,c as 3x LE uint32][h*w*c raw HWC BGR uint8]``."""
    lmdb = pytest.importorskip("lmdb")
    from utils.utils_video import RAW_LMDB_MAGIC

    root = tmp_path_factory.mktemp("multiscale")
    db = root / "gt_raw.lmdb"
    h, w, c, clip = 720, 1280, 3, "000"
    rng = np.random.default_rng(0)

    env = lmdb.open(str(db), map_size=int(4e9))
    with env.begin(write=True) as txn:
        for i in range(NUM_FRAME):
            # Blocky rather than pure noise so the antialiased resize and the
            # variance re-roll operate on something with structure.
            base = rng.integers(0, 256, (h // 8, w // 8, c), dtype=np.uint8)
            img = np.ascontiguousarray(np.repeat(np.repeat(base, 8, 0), 8, 1))
            header = RAW_LMDB_MAGIC + np.array([h, w, c], dtype="<u4").tobytes()
            txn.put(f"{clip}/{i:08d}".encode("ascii"), header + img.tobytes())
    env.close()

    meta = root / "meta_info_file.txt"
    meta.write_text(f"{clip} {NUM_FRAME} ({h},{w},{c}) 0\n")
    return str(db), str(meta)


def _dataset(raw_gt_lmdb, **overrides):
    from data.dataset_video_train import VideoRecurrentTrainDataset

    db, meta = raw_gt_lmdb
    opt = {
        "dataroot_gt": db,
        "meta_info_file": meta,
        "io_backend": {"type": "lmdb"},
        "raw_lmdb": True,
        "num_frame": NUM_FRAME,
        "tsf": TSF,
        "n_interp_samples": N_INTERP,
        "lq_patch": LQ_PATCH,
        "use_hflip": True,
        "use_rot": True,
        "name": "Adobe240",
        "test_mode": False,
    }
    opt.update(overrides)
    return VideoRecurrentTrainDataset(opt)


def test_dataset_reads_no_lq_and_infers_the_gt_crop(raw_gt_lmdb):
    dataset = _dataset(raw_gt_lmdb)
    assert dataset.multi_scale
    for step, scale in enumerate([1.2, 2.0, 2.5, 3.7, 4.0]):
        item = dataset[(step % len(dataset), scale)]
        expected = round(LQ_PATCH * scale)
        assert "L" not in item, "arbitrary-scale mode must not read any LR from disk"
        assert item["H"].shape[-2:] == (expected, expected)
        assert int(item["gt_size"]) == expected
        assert int(item["lq_patch"]) == LQ_PATCH


def test_sparse_gt_layout_and_anchor_positions(raw_gt_lmdb):
    dataset = _dataset(raw_gt_lmdb)
    item = dataset[(0, 3.0)]
    # anchors + n_interp mids per gap
    assert item["H"].shape[0] == T_IN + N_INTERP * (T_IN - 1)
    assert item["lq_sel"].shape[0] == T_IN
    assert item["interp_k"].shape == (T_IN - 1, N_INTERP)
    # The anchors must be the frames the LR is built from, and H is sparse, so
    # their positions inside H are not a fixed stride.
    assert item["lq_sel"].tolist() == sorted(set(item["lq_sel"].tolist()))


def test_missing_scale_is_refused(raw_gt_lmdb):
    """A bare index means the loader was built without MultiScaleBatchSampler.
    Failing loudly matters: the alternative is training at one accidental scale."""
    dataset = _dataset(raw_gt_lmdb)
    with pytest.raises(ValueError, match="without a scale"):
        _ = dataset[0]


def test_crop_larger_than_the_frame_is_refused(raw_gt_lmdb):
    dataset = _dataset(raw_gt_lmdb, lq_patch=512)  # 512 * 4 = 2048 > 1280
    with pytest.raises(ValueError, match="smaller than the crop"):
        _ = dataset[(0, 4.0)]


def test_arbitrary_scale_requires_per_sample_interp(raw_gt_lmdb):
    """Without per-sample interp the anchor frames are not guaranteed to be loaded,
    so the LR would be built from the wrong frames. Refuse rather than corrupt."""
    with pytest.raises(ValueError, match="per-sample temporal"):
        _dataset(raw_gt_lmdb, n_interp_samples=None)


def test_lr_synthesis_closes_the_geometry(raw_gt_lmdb):
    """The full training geometry: GT crop -> LR of exactly lq_patch, and a
    realised scale that maps one back onto the other."""
    pytest.importorskip("vsr")
    from vsr.utils.matlab_functions import imresize_gpu

    dataset = _dataset(raw_gt_lmdb)
    for scale in (1.2, 1.87, 2.5, 3.42, 4.0):
        item = dataset[(0, scale)]
        gt = item["H"].unsqueeze(0)
        g = gt.shape[-2]
        anchors = gt.index_select(1, item["lq_sel"])
        lr = imresize_gpu(anchors, LQ_PATCH / g)
        assert lr.shape[-2:] == (LQ_PATCH, LQ_PATCH), (scale, g, lr.shape)
        assert torch.isfinite(lr).all()
        s_eff = g / LQ_PATCH
        assert round(LQ_PATCH * s_eff) == g
