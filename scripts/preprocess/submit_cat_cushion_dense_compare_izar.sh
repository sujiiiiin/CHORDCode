#!/bin/bash
#SBATCH --job-name=chord_dense_compare
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/chord_dense_compare-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_HOME="$CONDA_ENV_DIR"
# Usage: sbatch this_script.sh DIRECT_MODEL [OPTIMIZED_MODEL] [OUTPUT_DIR]
# DIRECT_MODEL can be the default/ directory produced by direct_debug.
# Example: sbatch scripts/preprocess/submit_cat_cushion_dense_compare_izar.sh trained/cat_with_cushion/outputs_preprocess/direct_uv_3143038
DIRECT_MODEL="${1:?Usage: sbatch $0 DIRECT_MODEL [OPTIMIZED_MODEL] [OUTPUT_DIR]}"
OPTIMIZED_MODEL="${2:-trained/cat_with_cushion}"
OUT="${3:-trained/cat_with_cushion/outputs_preprocess/preprocess_comparison/run_${SLURM_JOB_ID}}"
export OUT
# Optional third panel for shape ablations, without another submit script.
VARIANTS=(direct optimized)
if [[ -n "${THIRD_MODEL:-}" ]]; then VARIANTS+=(third); fi
for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == direct ]]; then
    MODEL="$DIRECT_MODEL"
  elif [[ "$variant" == third ]]; then
    MODEL="$THIRD_MODEL"
  else
    MODEL="$OPTIMIZED_MODEL"
  fi
  "$CONDA_ENV_DIR/bin/python" scripts/evaluation/render_scene_static.py \
    -m "$MODEL" --mesh_source_path data/cat_with_cushion \
    --frame_num 1 --image_width 832 --image_height 480 \
    --cam_radius 1.8 --elev_l 10 --azimuth_step 3 --output_dir "$OUT/$variant"
done

# Sort by numeric azimuth; each of the 120 views is one frame at 15 fps.
"$CONDA_ENV_DIR/bin/python" - <<'PY'
import os
import subprocess
import tempfile
from pathlib import Path
import imageio_ffmpeg

out = Path(os.environ['OUT']).resolve()
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

def encode(args, destination):
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', *args,
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(destination)], check=True)

variants = ['direct', 'optimized'] + (['third'] if os.environ.get('THIRD_MODEL') else [])
with tempfile.TemporaryDirectory(prefix='chord_compare_') as tmp:
    for variant in variants:
        sequence = Path(tmp) / variant
        sequence.mkdir()
        for index, angle in enumerate(range(0, 360, 3)):
            source = out / variant / f'frame_{angle}_0.png'
            if not source.is_file():
                raise FileNotFoundError(source)
            (sequence / f'{index:03d}.png').symlink_to(source)
        destination = out / f'{variant}.mp4'
        encode(['-framerate', '15', '-i', str(sequence / '%03d.png')], destination)

destination = out / 'preprocess_comparison.mp4'
inputs = [arg for variant in variants for arg in ('-i', str(out / f'{variant}.mp4'))]
stack = ''.join(f'[{i}:v]' for i in range(len(variants)))
encode([*inputs, '-filter_complex', f'{stack}hstack=inputs={len(variants)}[v]', '-map', '[v]'], destination)
for name, size in [(v, (832, 480)) for v in variants] + [('preprocess_comparison', (832 * len(variants), 480))]:
    frames = imageio_ffmpeg.read_frames(str(out / f'{name}.mp4'))
    metadata = next(frames)
    count = sum(1 for _ in frames)
    if count != 120 or metadata['size'] != size:
        raise RuntimeError(f'Unexpected video format: {name}: {count}, {metadata}')
print(f'Comparison complete (panel order: {variants}): {destination}')
PY
