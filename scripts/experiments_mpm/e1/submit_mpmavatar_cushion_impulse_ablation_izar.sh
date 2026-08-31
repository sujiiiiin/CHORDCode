#!/bin/bash
#SBATCH --job-name=mpm_impulse_ablate
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/mpmavatar_impulse_ablation-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
ROOT="trained/cat_with_cushion/mpmavatar_e1_impulse_ablation"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_impulse_ablation_${SLURM_JOB_ID}"

run_case() {
  local name="$1" force="$2" force_duration="$3" youngs_modulus="$4"
  "$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e1/run_mpmavatar_cushion_impulse.py \
    --output-dir "$ROOT/$name" --force-magnitude "$force" \
    --force-duration "$force_duration" --youngs-modulus "$youngs_modulus" \
    --duration 0.06 --output-frames 301 --diagnostic-steps 0 --device cuda:0
}

run_case baseline       0.1 0.002 1200
run_case stronger_force 1.0 0.002 1200
run_case longer_force   0.1 0.020 1200
run_case softer_cushion 0.1 0.002 120
