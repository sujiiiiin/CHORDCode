#!/bin/bash
#SBATCH --job-name=mpmavatar_grid_conv
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/mpmavatar_grid_conv-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
OUTPUT_ROOT="trained/cat_with_cushion/mpmavatar_e1_full_convergence"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_grid_conv_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/mpmavatar_grid_conv_mpl_${SLURM_JOB_ID}"

for setting in "40 0.04 0.0004" "80 0.02 0.0002" "120 0.0133333333 0.0001333333333" "160 0.01 0.0001"; do
  read -r resolution pitch dt <<< "$setting"
  "$CONDA_ENV_DIR/bin/python" \
    scripts/experiments_mpm/e1/run_mpmavatar_sphere_cushion_smoke.py \
    --scene-dir data/cat_with_cushion \
    --output-dir "$OUTPUT_ROOT/grid_${resolution}" \
    --device cuda:0 \
    --pitch "$pitch" \
    --grid-resolution "$resolution" \
    --grid-limit 1.6 \
    --dt "$dt"
done
