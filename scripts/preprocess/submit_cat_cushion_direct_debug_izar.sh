#!/bin/bash
#SBATCH --job-name=chord_direct_preprocess
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/chord_direct_preprocess-%j.out
#SBATCH --chdir=/home/ydu/code/CHORDCode

set -euo pipefail
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/scratch/izar/ydu/.conda/envs/chord0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PWD/scripts/preprocess:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DIRECT_OUTPUT="${DIRECT_OUTPUT:-$PWD/trained/cat_with_cushion/outputs_preprocess/direct_preprocess_debug_${SLURM_JOB_ID}}"

# This conversion is CPU based; GPU allocation follows the Izar debug partition.
# No training iterations or Blender lighting bake are involved.
"$CONDA_ENV_DIR/bin/python" - <<'PY'
import json
import os
import time
from pathlib import Path
import numpy as np
from plyfile import PlyData
import convert_mesh_blender as convert

output = Path(os.environ['DIRECT_OUTPUT'])
output.mkdir(parents=True, exist_ok=False)
records = []
ablation = os.environ.get('SHAPE_ABLATION', '0') == '1'
stages = [('legacy', 30000, 100000, 8), ('surface', 30000, 100000, 8), ('surface_small', 30000, 100000, 8)] if ablation else [('smoke', 5000, 20000, 2), ('default', 30000, 100000, 8)]
if os.environ.get('SHAPE_ABLATION') == 'robust':
    stages = [('surface_small', 30000, 100000, 8), ('surface_robust', 30000, 100000, 8)]
    ablation = True
for label, target, maximum, rounds in stages:
    args = convert.parse_args([
        '--mesh_source_path', 'data/cat_with_cushion',
        '--model_path', str(output / label), '--convert_all',
        '--target_vertices', str(target), '--max_vertices', str(maximum),
        '--max_subdivide_iter', str(rounds), '--save_densified_mesh',
    ])
    if ablation and label != 'legacy':
        args.shape_mode = 'surface_robust' if label == 'surface_robust' else 'surface'
        args.tangent_factor = 0.65 if label in ('surface_small', 'surface_robust') else 1.0
    start = time.perf_counter()
    source = Path(args.mesh_source_path).resolve()
    reference = convert.maybe_get_scene_normalization_ref(source, args.self_norm)
    for job in convert.collect_mesh_jobs(args, source):
        begin = time.perf_counter()
        convert.convert_single_mesh(args, job, reference)
        elapsed = time.perf_counter() - begin
        ply = PlyData.read(job.output_root / 'point_cloud/iteration_0/point_cloud.ply')['vertex'].data
        mesh = convert.load_mesh_as_trimesh(job.output_root / 'densified_mesh.ply')
        xyz = np.stack([ply[k] for k in ('x', 'y', 'z')], axis=1)
        assert len(xyz) == len(mesh.vertices)
        assert np.allclose(xyz, mesh.vertices, atol=1e-6)
        assert all(np.isfinite(ply[k]).all() for k in ply.dtype.names)
        adjacency = json.loads((job.output_root / 'connected_vertices.json').read_text())
        assert len(adjacency) == len(xyz)
        record = dict(stage=label, object=job.display_name, seconds=elapsed,
                      gaussians=len(xyz), faces=len(mesh.faces), validation='passed')
        records.append(record)
        print('[RESULT]', json.dumps(record), flush=True)
        (output / 'summary.json').write_text(json.dumps(records, indent=2))
    print(f'[STAGE] {label} total_seconds={time.perf_counter() - start:.3f}', flush=True)
print(f'[DONE] {output}', flush=True)
PY
