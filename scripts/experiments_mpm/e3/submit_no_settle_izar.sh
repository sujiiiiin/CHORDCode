#!/bin/bash
#SBATCH --job-name=e3_no_settle
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ydu/code/CHORDCode
#SBATCH --output=logs/e3_no_settle-%j.out
set -euo pipefail
module purge
module load gcc/11.3.0
module load cuda/11.8.0
module load ffmpeg/4.4.1-h264
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/e3_no_settle_${SLURM_JOB_ID}"
/scratch/izar/ydu/.conda/envs/mpmavatar/bin/python scripts/experiments_mpm/e3/run_no_settle_mpm.py \
  --settle-delay 0 --output-dir "trained/cat_with_cushion/outputs_mpmavatar_e3/no_settle_${SLURM_JOB_ID}" "$@"
