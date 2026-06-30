#!/bin/bash
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --time=100:00:00

cd ${SLURM_SUBMIT_DIR} 

# clean modules
module purge

# load modules
# The .venv below is layered on the pytorch-gpu module (system-site-packages),
# so these MUST be loaded first — otherwise torch/MKL, basicsr, vsr and
# idr_accelerate are all missing in the job. Mirrors the interactive setup.
module load arch/h100
module load pytorch-gpu/py3/2.8.0

# Uncomment to load custom packages
source /linkhome/rech/gennip01/ura93tx/storage/.venv/bin/activate

# set WANDB offline mode
export WANDB_MODE=offline

OPT=$1

# execution
srun idr_accelerate /linkhome/rech/gennip01/ura93tx/Research/KAIR/main_train_elvsr.py --opt "$OPT"
