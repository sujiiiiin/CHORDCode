#!/bin/bash
#SBATCH --job-name=cat_mpm_debug
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/cat_mpm_debug-%j.out

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
export XDG_CACHE_HOME="/tmp/chord_mpm_debug_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/chord_mpm_debug_mpl_${SLURM_JOB_ID}"

run_debug() {
  local name="$1"
  shift
  echo "[E1-debug] Running $name"
  "$PYTHON_BIN" scripts/experiments/run_cat_cushion_mpm_e1.py \
    --scene-dir data/cat_with_cushion \
    --output-dir "$OUTPUT_ROOT/$name" \
    --device cuda:0 \
    --duration 1.0 \
    --extra-recovery-duration 4.0 \
    --output-frames 251 \
    --controls contact_friction0 \
    "$@"
}

# A: isolate cumulative explicit damping.
run_debug damping_09995 --damping 0.9995
run_debug damping_09999 --damping 0.9999
run_debug damping_1 --damping 1.0

# B: isolate the amount of welded bottom material with explicit damping disabled.
run_debug fixed_1layer --damping 1.0 --fixed-bottom-layers 1
run_debug fixed_3layers --damping 1.0 --fixed-bottom-layers 3

# C: isolate post-G2P particle projection while retaining grid contact.
run_debug no_particle_projection --damping 1.0 --fixed-bottom-layers 1 --no-particle-projection

# D: test whether the residual is primarily a time-integration artifact.
run_debug half_dt --damping 1.0 --fixed-bottom-layers 1 --dt 0.0001
