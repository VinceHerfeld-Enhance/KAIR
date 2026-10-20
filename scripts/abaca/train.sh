#!/usr/bin/env bash
# Train one arm on all GPUs of the current OAR job (DDP via torchrun). Resumes automatically from the
# last checkpoint in experiments/basicvsrpp_abl/<task>/models, so a killed job is simply resubmitted.
#
#     bash scripts/abaca/train.sh options/basicvsrpp_abl/C_2round.json
set -euo pipefail
OPT="${1:?usage: train.sh <option.json>}"
source "$(dirname "$0")/env.sh"

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
# KAIR's option parser exports CUDA_VISIBLE_DEVICES from gpu_ids, so gpu_ids must name the GPUs OAR
# gave this job; rewrite them (and nothing else) into a per-job copy of the option file.
mkdir -p logs/opts
RUN_OPT="logs/opts/$(basename "$OPT" .json).${OAR_JOB_ID:-local}.json"
python - "$OPT" "$RUN_OPT" "$NGPU" <<'EOF_PY'
import json, os, sys
src, dst, ngpu = sys.argv[1], sys.argv[2], int(sys.argv[3])
opt = json.load(open(src))
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
opt["gpu_ids"] = [int(x) for x in visible.split(",")] if visible else list(range(ngpu))
assert len(opt["gpu_ids"]) == ngpu, (opt["gpu_ids"], ngpu)
bs = opt["datasets"]["train"]["dataloader_batch_size"]
assert bs % ngpu == 0, f"batch {bs} not divisible by {ngpu} GPUs"
json.dump(opt, open(dst, "w"), indent=2)
print(f"[train] {opt['task']}: gpu_ids={opt['gpu_ids']}, batch {bs} total ({bs // ngpu}/GPU)")
EOF_PY

PORT=$((20000 + RANDOM % 20000))
torchrun --standalone --nproc_per_node="$NGPU" --master_port="$PORT" main_train_vrt.py --opt "$RUN_OPT" --dist True
