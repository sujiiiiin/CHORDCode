#!/bin/bash
#SBATCH --job-name=e3_interface
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/e3_interface-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode
set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export MPLCONFIGDIR="/tmp/e3_interface_mpl_${SLURM_JOB_ID}"
"$CONDA_ENV_DIR/bin/python" scripts/experiments_mpm/e3/check_refinement_interface.py \
  --output-dir "trained/cat_with_cushion/outputs_mpmavatar_e3/interface_${SLURM_JOB_ID}" "$@"
