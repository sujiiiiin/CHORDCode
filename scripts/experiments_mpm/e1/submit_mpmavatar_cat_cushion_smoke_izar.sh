#!/bin/bash
#SBATCH --job-name=mpm_cat_cushion
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/mpmavatar_cat_cushion-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_cat_cushion_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/mpmavatar_cat_cushion_mpl_${SLURM_JOB_ID}"

"$CONDA_ENV_DIR/bin/python" \
  scripts/experiments_mpm/e1/run_mpmavatar_cat_cushion_smoke.py \
  --scene-dir data/cat_with_cushion \
  --output-dir trained/cat_with_cushion/mpmavatar_e1_cat_cushion_smoke \
  --device cuda:0
