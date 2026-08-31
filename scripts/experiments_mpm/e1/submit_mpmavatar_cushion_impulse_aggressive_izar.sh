#!/bin/bash
#SBATCH --job-name=mpm_impulse_aggr
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/mpmavatar_impulse_aggressive-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
ROOT="trained/cat_with_cushion/mpmavatar_e1_impulse_aggressive"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_impulse_aggressive_${SLURM_JOB_ID}"

run_case() {
  local name="$1" force="$2" force_duration="$3"
  "$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e1/run_mpmavatar_cushion_impulse.py \
    --output-dir "$ROOT/$name" --force-magnitude "$force" \
    --force-duration "$force_duration" --youngs-modulus 120 \
    --duration 0.1 --output-frames 501 --diagnostic-steps 0 --device cuda:0
}

run_case force_5_duration_10ms 5.0 0.01
run_case force_10_duration_20ms 10.0 0.02
