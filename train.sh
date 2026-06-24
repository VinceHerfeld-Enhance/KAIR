#!/bin/bash
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH -C h100
#SBATCH --time=99:00:00

cd ${SLURM_SUBMIT_DIR} 

# clean modules
module purge

# load modules
source /linkhome/rech/gennip01/ura93tx/storage/.venv/bin/activate

# set WANDB offline mode
export WANDB_MODE=offline

OPT=$1

# execution
srun python3 /linkhome/rech/gennip01/ura93tx/Research/KAIR/main_train_elvsr.py --opt "$OPT"
