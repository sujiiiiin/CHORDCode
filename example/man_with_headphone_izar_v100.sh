#!/bin/bash
#SBATCH --job-name=man_headphone_izar
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/man_headphone_izar-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

# Izar/V100 adaptation of example/man_with_headphone.sh.
# Run this inside a Slurm allocation with one V100 and at least 64 GB system RAM.

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
EX_NAME="${EX_NAME:-man_with_headphone_izar_v100}"
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
  --add_cp_num 1500 \
  --mult_rot_way 0 \
  --back_iter 100 \
  --add_cp_layer_iter 300 \
  --add_cp_objs obj_2 \
  --init_voxel_size 0.015 \
  --batch_size 1 \
  --lambda_arap 2.4 \
  --invert_bg_prob -0.5 \
  --azim_l 45.0 \
  --azim_r 315.0 \
  --ref_azim 120.0 \
  --ref_cam_radius 1.2 \
  --elev_l 0.0 \
  --elev_r 30.0 \
  --cam_radius 1.2 \
  --resample_timestep \
  --cp_num 60 \
  --lambda_dis_time 2.0 \
  --time_loss_landmarks 6.0 6.0 6.0 4.0 1.0 \
  --last_cfg_scale 12.0 \
  --init_cfg_scale 25.0 \
  --frame_num 41 \
  --save_interval 500 \
  --mesh_source_path data/man_with_headphone/ \
  --model_path trained/man_with_headphone \
  --task i2v-A14B \
  --size 416*240 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --prompt "Move both hands to the headband. Place thumbs on the outer band and fingers under the inner band near each earcup. Grip both sides, lift the headphones off the table, then pull both hands outward evenly to open the headband." \
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
