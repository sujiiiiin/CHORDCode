#!/bin/bash
#SBATCH --job-name=cat_cushion_stage2
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ydu/code/CHORDCode
#SBATCH --output=logs/cat_cushion_stage2_debug-%j.out
set -euo pipefail
module purge
module load gcc/11.3.0
module load ffmpeg/4.4.1-h264
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/cat_cushion_stage2_${SLURM_JOB_ID}"
CHORD_PYTHON="${CHORD_PYTHON:-/scratch/izar/ydu/.conda/envs/chord0/bin/python}"
MPM_PYTHON="${MPM_PYTHON:-$CHORD_PYTHON}"
export CUDA_HOME="/scratch/izar/ydu/.conda/envs/chord0"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_MODULE_LOADING=LAZY
"$CHORD_PYTHON" train.py --stage 2 --cat_with_cushion \
  --chord-python "$CHORD_PYTHON" --mpm-python "$MPM_PYTHON" \
  --stage1-model-dir trained/cat_with_cushion \
  --stage1-experiment cat_with_cushion_izar_v100 --stage1-checkpoint 3000 \
  --mpm-output-dir "trained/cat_with_cushion/stage2_debug_${SLURM_JOB_ID}"
