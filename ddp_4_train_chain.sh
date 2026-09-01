#!/bin/bash
#SBATCH --partition=gpu_p6
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=19:59:00

# One link in a chain of jobs (see submit_chain.sh). Uses qos_gpu_h100-t3
# instead of ddp_4_train.sh's qos_gpu_h100-t4: t3 has scheduling priority 50
# vs t4's 45, and t4 is meant for long, low-priority, best-effort jobs, which
# is why jobs sat pending there. t3 caps a single job at 20h, so this relies
# on being resubmitted as a chain to reach total_iter.

cd ${SLURM_SUBMIT_DIR}

# clean modules
module purge

# load modules
module load arch/h100
module load pytorch-gpu/py3/2.8.0

# Uncomment to load custom packages
source /linkhome/rech/gennip01/ura93tx/storage/.venv/bin/activate

# set WANDB offline mode
export WANDB_MODE=offline

# Reduces allocator fragmentation from cycling ~180 distinct crop sizes (closes
# a ~17GB reserved-vs-allocated gap; see stvsr_speedup.out). Allocator-only,
# no numerical effect.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OPT=$1

# Skip as a no-op if a previous link in the chain already reached total_iter.
# main_train_elvsr.py resumes from the last checkpoint on its own either way;
# this just avoids burning a queue slot on a run that would do nothing.
DONE=$(python3 - "$OPT" <<'PY'
import json, sys, glob, os, re

opt = json.load(open(sys.argv[1]))
models_dir = opt["path"]["models"]
total_iter = opt["train"]["total_iter"]

iters = []
for f in glob.glob(os.path.join(models_dir, "*_G.pth")):
    m = re.match(r".*/(\d+)_G\.pth$", f)
    if m:
        iters.append(int(m.group(1)))

print("yes" if iters and max(iters) >= total_iter else "no")
PY
)

if [ "$DONE" = "yes" ]; then
    echo "Latest checkpoint already reached total_iter, skipping this chain link."
    exit 0
fi

# execution
srun idr_accelerate /linkhome/rech/gennip01/ura93tx/Research/KAIR/main_train_elvsr.py --opt "$OPT"
