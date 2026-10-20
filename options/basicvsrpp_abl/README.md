# BasicVSR++ with 2 branches instead of 4

**Question.** BasicVSR++'s second propagation round (branches 3–4) adds +0.31 dB (paper Table 3:
32.08 → 32.39 on REDS4). Hypothesis: it compensates for two weaknesses of round-1 alignment,
(1) **bilinear resampling blur** in the DCN and (2) **first-order, pairwise motion** (SpyNet
fails on fast motion). Fix both inside one round and two branches should match four.

Network: `VSR/src/vsr/research_models/basicvsrpp_ablation.py` (`net_type: "basicvsrpp_abl"`).
With default options it *is* BasicVSR++: same state_dict keys, and its output matches basicsr's
`BasicVSRPlusPlus` exactly (`VSR/test/test_basicvsrpp_ablation.py`). The arms differ only in `netG`.
Data, schedule and head are identical, generated from one base by `make_configs.py` (edit it and
regenerate; don't hand-edit the JSONs).

## Arms

| Arm | Branches | Change | Tests |
|---|---|---|---|
| `A_1round` | b1 f1 | — | floor (paper: 32.08) |
| `C_2round` | b1 f1 b2 f2 | — | **target** (paper: 32.39) |
| `A5_1round_deep` | b1 f1 | `num_blocks` 23 | capacity control: FLOPs and params matched to C (833 vs 830 GFLOPs/frame, 7.23 M vs 7.32 M) |
| `B1_snap` | b1 f1 | snap DCN | does blur-free resampling replace round 2? |
| `B2_rqfp` | b1 f1 | RQFP flows | does acceleration-aware motion replace round 2? |
| `B3_snap_rqfp` | b1 f1 | both | **the method** |
| `A3_3branch` | b1 f1 b2 | — | is round 2's gain in the second *backward* pass? |
| `D_2round_snap` | 4 | snap DCN | does snap help even with 2 rounds? (2×2 with A, B1, C) |
| `B1_snap_window` | b1 f1 | snap, `max_residue_magnitude` 0 | IART-style local window vs multi-location gather |
| `B1_snap_ste` | b1 f1 | snap, straight-through grads | gradient mode alternative |
| `B2_rqfp_spynet` | b1 f1 | RQFP as SpyNet warm start | fast-motion SpyNet failures |
| `B2_learned` | b1 f1 | RQFP + learned flow residual | is learning needed beyond the closed form? |
| `pilot_snap_mask_only`, `pilot_snap_ste` | b1 f1 | snap, 20k iters | do snapped offsets learn at all? |

Options in brief:
- **Snap DCN.** Offsets are predicted exactly as in BasicVSR++ (flow + 10·tanh residual). The DCN
  then samples at `round(offset)`, which is an exact nearest-neighbour fetch through the same
  torchvision op. The fractional remainder is sine-encoded and, together with each tap's kernel
  position, an MLP turns it into an additive modulation logit (zero-initialised). The tap
  position is what lets the weights favour the samples nearest the true sub-pixel point; in the
  window arm every tap shares one fraction. In `mask_only` mode the loss reaches the offsets only
  through that fraction, as in IART. In `ste` mode the fetch also gets a straight-through
  bilinear gradient.
- **RQFP.** A least-squares fit of `f_k = v·k + a·k²/2` over the ±1/±2 neighbours, re-evaluated
  at k = ±1. It has no parameters. Plain QVI (2 observations) would return the input flow
  unchanged.

Cost per frame at 180×320 (`VSR/scripts/count_basicvsrpp_ablation_cost.py`): 1 round 561 GFLOPs,
2 rounds 830, snap +5, rqfp +0, rqfp_spynet +64. Round 2 is 32% of the FLOPs but half the
*sequential* propagation steps.

## Setup on Abaca (once)

1. **Code.** Check out this KAIR branch and the matching VSR branch side by side
   (`<dir>/KAIR`, `<dir>/VSR`). `scripts/abaca/env.sh` puts `../VSR/src` on `PYTHONPATH`;
   override it with `VSR_SRC`.
2. **Venv.** Needs torch + torchvision with CUDA, `basicsr`, `wandb`, `opencv-python`, `lmdb`,
   `requests`. No compiled extension is needed. The default is `~/venv-py312`; override with `VENV`.
3. **SpyNet weights.** Compute nodes may be offline, so fetch them on the frontend:
   `mkdir -p model_zoo/basicvsrpp && curl -L -o model_zoo/basicvsrpp/spynet_sintel_final-3d2a1287.pth https://github.com/JingyunLiang/RVRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth`
4. **REDS.** Download train_sharp + train_sharp_bicubic, run
   `scripts/data_preparation/regroup_reds_dataset.py`, then build the LMDBs with
   `scripts/data_preparation/create_lmdb.py` (the paths in `create_lmdb_for_reds` are hard-coded;
   edit them). Then link them so that these paths exist under KAIR:
   - `trainsets/REDS/train_sharp_with_val.lmdb`
   - `trainsets/REDS/train_sharp_bicubic_with_val.lmdb`
   - `testsets/REDS4/GT/{000,011,015,020}`
   - `testsets/REDS4/sharp_bicubic/{000,011,015,020}`
   REDS4 is excluded from training via `val_partition`.

## Running

```bash
cd KAIR
bash scripts/abaca/submit.sh pilot_snap_mask_only pilot_snap_ste           # 1. pilots (20k iters)
bash scripts/abaca/submit.sh A_1round C_2round A5_1round_deep \
                             B1_snap B2_rqfp B3_snap_rqfp                   # 2. tier 1
bash scripts/abaca/submit.sh A3_3branch D_2round_snap B1_snap_window \
                             B1_snap_ste B2_rqfp_spynet B2_learned          # 3. tier 2, as budget allows
```

Defaults: 4 GPUs per job (batch 8 total = 2 per GPU), 48 h walltime, 150k iterations.
Override with `GPUS`, `WALLTIME`, `PROPS` (an `oarsub -p` filter), `QUEUE`. Jobs resume from
their last checkpoint (every 10k iterations), so a killed or besteffort job is simply
resubmitted with the same command. wandb runs offline (project `BasicVSRpp_2branch`); sync with
`wandb sync`.

**Pilot gate.** After 20k iterations, evaluate both pilots (below) and look at the `offsets`
line. If `residual_px` stays near 0 in `mask_only`, the snapped offsets aren't learning: switch
B1/B3 to `snap_grad: "ste"` in `make_configs.py` and regenerate. If `ste` shows saturation
climbing toward 100%, keep `mask_only`.

## Evaluation

```bash
cd VSR
for arm in A_1round C_2round A5_1round_deep B1_snap B2_rqfp B3_snap_rqfp; do
  PYTHONPATH=src python scripts/analyze_basicvsrpp_ablation.py eval \
    --opt ../KAIR/options/basicvsrpp_abl/$arm.json \
    --ckpt ../KAIR/experiments/basicvsrpp_abl/bvpp_abl_$arm/models/150000_G.pth \
    --lq ../KAIR/testsets/REDS4/sharp_bicubic --gt ../KAIR/testsets/REDS4/GT --out results/$arm.json
done
PYTHONPATH=src python scripts/analyze_basicvsrpp_ablation.py compare results/*.json --reference A_1round
```

This prints REDS4 PSNR, PSNR binned by motion magnitude (a fixed SpyNet reference, identical for
every arm) with deltas against A, and per-branch offset diagnostics.

## Reading the outcome

| Result | Meaning |
|---|---|
| B3 ≥ C | 2 branches suffice: the paper |
| A5 ≈ C | round 2 was capacity; reframe as "the grid is unnecessary structure" |
| B1 ≈ C, D ≈ C | round 2 was compensating for resampling blur |
| B2 ≈ C | round 2 was compensating for first-order motion |
| A3 ≈ C | the gain sits in the second backward pass (round-1 backward never sees the past) |
| B gains concentrated in the top motion bins | confirms the mechanism (the figure) |
| B3 between A and C | add SSM-style confidence propagation next |
| B1_snap_window ≈ B1_snap | a local window is enough; multi-location gather isn't needed |

Caveat: these are 150k-iteration runs, so absolute numbers sit below the paper's 600k. Compare
arms with each other, and treat differences under about 0.05 dB as noise unless they hold up
across clips.
