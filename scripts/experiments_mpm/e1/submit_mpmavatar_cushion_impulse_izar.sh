#!/bin/bash
#SBATCH --job-name=mpmavatar_impulse
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/mpmavatar_impulse-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/tmp/mpmavatar_impulse_${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/mpmavatar_impulse_mpl_${SLURM_JOB_ID}"

for force in 0.01 0.1 1.0 10.0; do
  OUTPUT_DIR="trained/cat_with_cushion/mpmavatar_e1_impulse_ladder/force_${force}"
  "$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e1/run_mpmavatar_cushion_impulse.py \
    --output-dir "$OUTPUT_DIR" --force-magnitude "$force" --device cuda:0
  "$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e1/visualize_mpmavatar_cushion_impulse.py \
    --input-dir "$OUTPUT_DIR"
done
