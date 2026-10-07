#!/bin/bash
#SBATCH --job-name=preprocess_demos
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/preprocess_demos-%A_%a.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

# One GPU job per scene; original CHORD preprocessing defaults are preserved.
# Static Gaussian fitting only: this does not start Wan motion supervision.
SCENES=(
  cutting_board_with_croissant
  baseball_with_pillow
  tyre_with_compost_bag
  cutting_board_with_rubber_duck
  broom_with_trashbag
)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
SCENE_NAME="${SCENES[$TASK_ID]}"
REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="/tmp/chord_preprocess_${SLURM_JOB_ID}"

echo "Preprocessing scene: $SCENE_NAME (array task $TASK_ID)"
"$PYTHON_BIN" scripts/preprocess/train_gaussian_for_scene.py \
  --model_path "trained/$SCENE_NAME" \
  --mesh_source_path "data/$SCENE_NAME"
echo "Preprocessing completed: $SCENE_NAME"
