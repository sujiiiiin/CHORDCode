#!/bin/bash

set -euo pipefail

# Izar/V100 adaptation of example/assassin_with_eagle.sh.
# Run this inside a Slurm allocation with one V100 and at least 64 GB system RAM.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
EX_NAME="${EX_NAME:-assassin_with_eagle_izar_v100}"
ITERATIONS="${ITERATIONS:-3000}"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"

cd "$REPO_DIR"

"$PYTHON_BIN" train.py \
  --ex_name "$EX_NAME" \
  --add_cp_detach_first \
  --early_stop \
  --add_cp_voxel_scale 2.5 \
  --add_cp_num 1000 \
  --add_cp_layer_iter 300 \
  --add_cp_objs obj_2 \
  --mult_rot_way 0 \
  --back_iter 100 \
  --obj_num 3 \
  --init_voxel_size 0.015 \
  --lambda_ground 0.0 \
  --batch_size 1 \
  --lambda_arap 2.4 \
  --invert_bg_prob -0.5 \
  --azim_l 0.0 \
  --azim_r 360.0 \
  --ref_azim 60.0 \
  --elev_l -10.0 \
  --elev_r 40.0 \
  --cam_radius 1.7 \
  --ref_cam_radius 1.7 \
  --resample_timestep \
  --cp_num 60 \
  --lambda_dis_time 2.4 \
  --time_loss_landmarks 6.0 6.0 6.0 4.0 1.0 \
  --last_cfg_scale 12.0 \
  --init_cfg_scale 25.0 \
  --frame_num 41 \
  --save_interval 500 \
  --mesh_source_path data/assassin_with_eagle/ \
  --model_path trained/assassin_with_eagle \
  --task i2v-A14B \
  --size 416*240 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --prompt "The man extends his forearm horizontally and steady, inviting the eagle to glide in, flare its wings, lower its talons, and perch." \
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
