#!/bin/bash
#SBATCH --job-name=cat_contact_p0
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cat_contact_p0-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"

"$PYTHON_BIN" scripts/experiments/diagnose_cat_cushion_contact.py \
  --scene-dir data/cat_with_cushion \
  --model-dir trained/cat_with_cushion \
  --experiment cat_with_cushion_izar_v100 \
  --checkpoint 3000 \
  --frame-count 41 \
  --foot-quantile 0.12 \
  --surface-samples 50000 \
  --window-padding 2 \
  --seed 0
