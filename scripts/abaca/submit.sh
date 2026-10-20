#!/usr/bin/env bash
# Submit BasicVSR++ study arms as OAR jobs on Abaca (one job per arm).
#
#     bash scripts/abaca/submit.sh pilot_snap_mask_only pilot_snap_ste
#     bash scripts/abaca/submit.sh A_1round C_2round A5_1round_deep B1_snap B2_rqfp B3_snap_rqfp
#     GPUS=8 WALLTIME=48:00:00 PROPS="gpu_mem >= 40000" bash scripts/abaca/submit.sh C_2round
#
# GPUS must divide the batch (8). PROPS goes to `oarsub -p` (check property names with `oarnodes`);
# QUEUE=besteffort works too: jobs resume from their last checkpoint when resubmitted.
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v oarsub >/dev/null || { echo "oarsub not found — run on an Abaca frontend"; exit 1; }
[[ $# -gt 0 ]] || { sed -n '2,9p' "$0"; exit 1; }

GPUS="${GPUS:-4}"
WALLTIME="${WALLTIME:-48:00:00}"
mkdir -p logs
for arm in "$@"; do
  opt="options/basicvsrpp_abl/$arm.json"
  [[ -f "$opt" ]] || { echo "no such arm: $opt"; exit 1; }
  args=(-n "bvpp-$arm" -l "host=1/gpu=$GPUS,walltime=$WALLTIME"
        -O "logs/bvpp-$arm.%jobid%.out" -E "logs/bvpp-$arm.%jobid%.err")
  [[ -n "${PROPS:-}" ]] && args+=(-p "$PROPS")
  [[ -n "${QUEUE:-}" ]] && args+=(-q "$QUEUE")
  oarsub "${args[@]}" "bash $PWD/scripts/abaca/train.sh $PWD/$opt"
done
