"""Batch sampler that assigns one spatial scale per optimizer step.

Why a *batch* sampler and not a ``collate_fn``
----------------------------------------------
Arbitrary-scale C-STVSR training needs the LR input synthesised at a randomly
drawn spatial scale. MoTIF and BF-STVSR both do this in a ``collate_fn``, and
their docstrings say why it cannot go in ``__getitem__``: with ``num_workers > 1``
each worker would draw its own scale and the items in a batch would have
mismatched resolutions.

A ``collate_fn`` fixes that but pays for it — the scale is only known *after* the
workers have read their frames, so the dataset has to hand over whole
full-resolution frames and let the collate crop them (which is exactly what
``MoTIF/data/Adobe_arbitrary.py`` does, with ``cv2.imread`` on 720p frames). This
repo has a raw-uint8 LMDB path that reads only the crop's row band
(``utils_video.LmdbBackend.get_raw_crop``), and throwing that away would be a
large read-time and memory regression.

Putting the draw in the *batch sampler* keeps both properties: the scale travels
with the index, so the worker knows it before it touches pixels and can size its
crop up front — which also means the read volume shrinks with the scale instead
of always paying the largest-scale footprint.

Determinism, and why it is mandatory here
-----------------------------------------
Accelerate shards the *prepared loader*: every process iterates this same batch
sampler and keeps the batches at positions ``rank, rank + W, rank + 2W, ...``.
Two consequences, both load-bearing:

* The emitted sequence **must be identical on every rank**, or the per-rank
  slices are not a partition of the epoch and samples are silently dropped or
  duplicated. So the index permutation is seeded from ``(seed, epoch)`` only —
  never from anything rank-local.
* The scale is drawn per **step group** (``batch_idx // group_size``) rather than
  per batch, so all ``W`` batches making up one optimizer step share a scale.
  With ``group_size == world_size`` this means every rank sees the same scale at
  the same step, which keeps the ``(4/s)^2`` loss reweighting and the logged
  scale meaningful instead of rank-dependent.

Both properties also make a run reproducible: the scale at step ``k`` of epoch
``e`` is a pure function of ``(seed, e, k)``.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Sampler


class MultiScaleBatchSampler(Sampler):
    """Yield batches of ``(index, scale)`` with one scale per optimizer step.

    Args:
        num_samples: dataset length.
        batch_size: per-process batch size (the same value handed to the loader).
        scale_range: ``(lo, hi)`` for the continuous uniform draw. ``(1.2, 4.0)``
            matches v3's ``--scale-range``; MoTIF / BF-STVSR use ``(2, 4)``.
        group_size: how many consecutive batches share one scale. Set this to the
            world size so one optimizer step is single-scale across ranks. ``1``
            for single-process training.
        shuffle: permute indices each epoch (seeded, see module docstring).
        drop_last: drop a trailing partial batch. Default ``True``, matching the
            training loader — a short batch is harmless here but keeps step
            accounting uniform.
        seed: base seed for the permutation and the scale draw.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        scale_range: Tuple[float, float] = (1.2, 4.0),
        group_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        lo, hi = float(scale_range[0]), float(scale_range[1])
        if not (0 < lo <= hi):
            raise ValueError(f"scale_range must satisfy 0 < lo <= hi, got {scale_range}")
        self.scale_lo, self.scale_hi = lo, hi
        self.group_size = max(1, int(group_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the epoch. Accelerate calls this when it is present."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def scale_for_step(self, step: int) -> float:
        """Scale for optimizer step ``step`` of the current epoch.

        A pure function of ``(seed, epoch, step)``, so it is identical on every
        rank and reproducible across restarts. Uses a private ``random.Random``
        rather than the global RNG so it cannot perturb (or be perturbed by) the
        crop/augmentation draws in the workers. The seed is mixed arithmetically
        rather than with ``hash()`` so it is deterministic by construction and
        does not depend on any interpreter hashing detail.
        """
        mixed = (self.seed * 1_000_003 + self.epoch * 10_007 + step) & 0x7FFFFFFF
        rng = random.Random(mixed)
        return rng.uniform(self.scale_lo, self.scale_hi)

    def __iter__(self) -> Iterator[List[Tuple[int, float]]]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            order = list(range(self.num_samples))

        n_batches = len(self)
        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            indices = order[start : start + self.batch_size]
            if not indices:
                continue
            scale = self.scale_for_step(batch_idx // self.group_size)
            yield [(int(i), scale) for i in indices]
