#!/bin/bash
#SBATCH --job-name=interactive_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -C h100
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=5     # request more CPUs → more RAM

srun --pty bash
