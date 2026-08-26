#!/bin/bash
#SBATCH --job-name=cat_contact_p0_vis
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/cat_contact_p0_vis-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export MPLCONFIGDIR="/tmp/chord_contact_p0_vis_mpl_${SLURM_JOB_ID}"

"$CONDA_ENV_DIR/bin/python" \
  scripts/experiments_mpm/p0/visualize_cat_cushion_contact_p0.py \
  --input-dir \
  trained/cat_with_cushion/cat_with_cushion_izar_v100/contact_diagnostics/deform_3000 \
  --fps 10
