#!/bin/bash
#SBATCH --job-name=cat_mpm_recovery
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cat_mpm_recovery-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export XDG_CACHE_HOME="/tmp/chord_mpm_recovery_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/chord_mpm_recovery_mpl_${SLURM_JOB_ID}"

"$PYTHON_BIN" scripts/experiments/run_cat_cushion_mpm_e1.py \
  --scene-dir data/cat_with_cushion \
  --output-dir trained/cat_with_cushion/mpm_e1/long_recovery \
  --device cuda:0 \
  --duration 1.0 \
  --extra-recovery-duration 4.0 \
  --output-frames 251
