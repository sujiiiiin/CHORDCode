#!/bin/bash
#SBATCH --job-name=e3_refine
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ydu/code/CHORDCode
#SBATCH --output=logs/e3_refine-%j.out
set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/e3_refine_${SLURM_JOB_ID}"
"$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e3/refine_cushion.py \
  --interface-dir "$1" --output-dir "trained/cat_with_cushion/outputs_mpmavatar_e3/refine_${SLURM_JOB_ID}"
