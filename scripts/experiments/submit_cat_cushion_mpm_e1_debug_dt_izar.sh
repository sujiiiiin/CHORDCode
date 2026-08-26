#!/bin/bash
#SBATCH --job-name=cat_mpm_debug_dt
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/cat_mpm_debug_dt-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
OUTPUT_ROOT="trained/cat_with_cushion/mpm_e1/debug"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export XDG_CACHE_HOME="/tmp/chord_mpm_debug_dt_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/chord_mpm_debug_dt_mpl_${SLURM_JOB_ID}"

run_dt() {
  local name="$1"
  shift
  echo "[E1-debug-dt] Running $name"
  "$PYTHON_BIN" scripts/experiments/run_cat_cushion_mpm_e1.py \
    --scene-dir data/cat_with_cushion \
    --output-dir "$OUTPUT_ROOT/$name" \
    --device cuda:0 \
    --duration 1.0 \
    --extra-recovery-duration 4.0 \
    --output-frames 251 \
    --controls contact_friction0 \
    --damping 1.0 \
    "$@"
}

# One-layer boundary: continue the existing dt=2e-4 and 1e-4 sequence.
run_dt fixed1_quarter_dt --fixed-bottom-layers 1 --dt 0.00005

# Default two-layer boundary: isolate dt without changing the original boundary.
run_dt fixed2_half_dt --fixed-bottom-layers 2 --dt 0.0001
run_dt fixed2_quarter_dt --fixed-bottom-layers 2 --dt 0.00005
