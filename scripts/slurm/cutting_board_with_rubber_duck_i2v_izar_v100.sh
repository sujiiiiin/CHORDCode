#!/bin/bash
#SBATCH --job-name=cutting_board_with_rubber_duck_i2v
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/cutting_board_with_rubber_duck_i2v-%j.out

#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail

# Based on scripts/slurm/assassin_with_eagle_izar_v100.sh.
# Supervision: Wan2.2 I2V-A14B, matching the original V100 template.
# Prompt loaded from data/cutting_board_with_rubber_duck/prompt.txt.
# Run this inside a Slurm allocation with one V100 and at least 64 GB system RAM.

REPO_DIR="$PWD"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
PYTHON_BIN="$CONDA_ENV_DIR/bin/python"
EX_NAME="${EX_NAME:-cutting_board_with_rubber_duck_i2v_izar_v100}"
ITERATIONS="${ITERATIONS:-3000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
CKPT_DIR="${CKPT_DIR:-./ckpts/Wan2.2-I2V-A14B}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$CONDA_ENV_DIR"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="7.0"

cd "$REPO_DIR"

mkdir -p logs

# Fail early if scratch cleanup removed any required model files.
"$PYTHON_BIN" - "$CKPT_DIR" <<'PY_CHECKPOINT'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
required = [root / "models_t5_umt5-xxl-enc-bf16.pth", root / "Wan2.1_VAE.pth",
            root / "google/umt5-xxl/tokenizer.json", root / "google/umt5-xxl/tokenizer_config.json"]
for model in ("high_noise_model", "low_noise_model"):
    index = root / model / "diffusion_pytorch_model.safetensors.index.json"
    required.extend([index, root / model / "config.json"])
    if index.is_file():
        required.extend(root / model / name for name in set(json.loads(index.read_text())["weight_map"].values()))
missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
if missing:
    sys.exit("Missing I2V checkpoint files:\n" + "\n".join(missing))
PY_CHECKPOINT

# Require all three final preprocessing outputs before starting motion training.
for obj in obj_0 obj_1 obj_2; do
  test -s "trained/cutting_board_with_rubber_duck/$obj/point_cloud/iteration_50000/point_cloud.ply"
done
PROMPT="$(cat data/cutting_board_with_rubber_duck/prompt.txt)"
test -n "$PROMPT"

"$PYTHON_BIN" train.py \
  --ex_name "$EX_NAME" \
  --add_cp_detach_first \
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
  --save_interval "$SAVE_INTERVAL" \
  --mesh_source_path data/cutting_board_with_rubber_duck/ \
  --model_path trained/cutting_board_with_rubber_duck \
  --task i2v-A14B \
  --size "416*240" \
  --ckpt_dir "$CKPT_DIR" \
  --prompt "$PROMPT" \
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
