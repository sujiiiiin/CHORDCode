#!/bin/bash
#SBATCH --job-name=mpm_impulse_hf
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/mpmavatar_impulse_highfreq-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_impulse_highfreq_${SLURM_JOB_ID}"

"$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e1/run_mpmavatar_cushion_impulse.py \
  --output-dir trained/cat_with_cushion/mpmavatar_e1_impulse_highfreq \
  --force-magnitude 0.1 --duration 0.02 --output-frames 101 \
  --diagnostic-steps 12 --device cuda:0
