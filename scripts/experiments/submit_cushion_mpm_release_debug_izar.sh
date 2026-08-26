#!/bin/bash
#SBATCH --job-name=mpm_release_debug
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mpm_release_debug-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
OUTPUT_ROOT="trained/cat_with_cushion/mpm_e1/debug_release"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export XDG_CACHE_HOME="/tmp/chord_mpm_release_${SLURM_JOB_ID}"

for dt_spec in "dt2e-4 0.0002" "dt1e-4 0.0001" "dt5e-5 0.00005"; do
  read -r name dt <<< "$dt_spec"
  "$PYTHON_BIN" scripts/experiments/run_cushion_mpm_release_debug.py \
    --output-dir "$OUTPUT_ROOT/$name" \
    --device cuda:0 \
    --dt "$dt" \
    --duration 5.0 \
    --damping 1.0 \
    --fixed-bottom-layers 2
done
