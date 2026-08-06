#!/bin/bash
#SBATCH --job-name=cat_cushion_izar
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/cat_cushion_izar-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

# Izar/V100 adaptation of example/cat_with_cushion.sh.
# Run this inside a Slurm allocation with one V100 and at least 64 GB system RAM.

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
EX_NAME="${EX_NAME:-cat_with_cushion_izar_v100}"
ITERATIONS="${ITERATIONS:-3000}"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"

cd "$REPO_DIR"

mkdir -p logs

"$PYTHON_BIN" train.py \
  --ex_name "$EX_NAME" \
  --add_cp_detach_first \
  --add_cp_voxel_scale 2.5 \
  --add_cp_num 300 \
  --add_cp_layer_iter 300 \
  --add_cp_objs obj_0 obj_2 \
  --mult_rot_way 0 \
  --back_iter 100 \
  --obj_num 3 \
  --lambda_ground 0.0 \
  --batch_size 1 \
  --lambda_arap 2.4 \
  --invert_bg_prob -0.5 \
  --azim_l 0.0 \
  --azim_r 360.0 \
  --ref_azim 60.0 \
  --elev_l 0.0 \
  --elev_r 40.0 \
  --cam_radius 1.3 \
  --ref_cam_radius 1.3 \
  --resample_timestep \
  --cp_num 60 \
  --lambda_dis_time 2.0 \
  --time_loss_landmarks 6.0 6.0 6.0 4.0 1.0 \
  --last_cfg_scale 12.0 \
  --init_cfg_scale 25.0 \
  --frame_num 41 \
  --save_interval 500 \
  --mesh_source_path data/cat_with_cushion/ \
  --model_path trained/cat_with_cushion \
  --task i2v-A14B \
  --size 416*240 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --prompt "The cat strolls across the cushion, the fabric dimpling beneath its weight and relaxing back to shape after every step." \
  --n_prompt "." \
  --use_tiny_vae \
  --enable_mmgp \
  --mmgp_profile 4.0 \
  --mmgp_transformer_budget 100 \
  --split_sds_backward \
  --split_render_backward \
  --split_render_chunk_size 21 \
  --log_run_time \
  --iterations "$ITERATIONS"
