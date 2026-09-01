#!/bin/bash
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --time=100:00:00

cd ${SLURM_SUBMIT_DIR} 

# clean modules
module purge

# load modules
module load arch/h100
module load pytorch-gpu/py3/2.8.0

# Uncomment to load custom packages
source /linkhome/rech/gennip01/ura93tx/storage/.venv/bin/activate

export PYTHONPATH=/linkhome/rech/gennip01/ura93tx/Research/VSR/src:$PYTHONPATH

# set WANDB offline mode
export WANDB_MODE=offline

OPT=$1

# execution
srun idr_accelerate /linkhome/rech/gennip01/ura93tx/Research/KAIR/main_train_elvsr.py --opt "$OPT"
