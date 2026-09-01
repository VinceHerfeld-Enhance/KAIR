#!/bin/bash
#SBATCH --partition=gpu_p6            
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --tasks-per-node=1
#SBATCH --nodes=1
#SBATCH -C h100
#SBATCH --hint=nomultithread
#SBATCH --time=99:59:00

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

# execution
srun idr_accelerate /linkhome/rech/gennip01/ura93tx/Research/KAIR/main_train_elvsr.py --opt "$OPT"
