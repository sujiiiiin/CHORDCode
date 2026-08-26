#!/usr/bin/env python3
"""P0: extract CHORD trajectories and diagnose cat/cushion proximity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import GaussianModel, Scene
from scene.dynamic_gaussian_model import DynamicGaussianModel
from utils.contact_diagnostics import (
    deform_surface_samples,
    frame_contact_metrics,
    infer_contact_windows,
    label_vertex_components,
    sample_surface_template,
    select_low_vertices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--model-dir", type=Path, default=Path("trained/cat_with_cushion"))
    parser.add_argument("--experiment", default="cat_with_cushion_izar_v100")
    parser.add_argument("--checkpoint", type=int, default=3000)
    parser.add_argument("--frame-count", type=int, default=41)
    parser.add_argument("--foot-quantile", type=float, default=0.12)
    parser.add_argument("--surface-samples", type=int, default=50_000)
    parser.add_argument("--contact-threshold", type=float, default=None)
    parser.add_argument("--min-contact-candidates", type=int, default=1)
    parser.add_argument("--window-padding", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_normalized_mesh(path: Path, scene_path: Path) -> tuple[np.ndarray, np.ndarray]:
    scene_mesh = trimesh.load(scene_path, force="mesh", process=False)
    mesh = trimesh.load(path, force="mesh", process=False)
    scene_min, scene_max = scene_mesh.bounds
    center = (scene_min + scene_max) * 0.5
    scale = 1.2 / np.max(scene_max - scene_min)
    vertices = (np.asarray(mesh.vertices, dtype=np.float32) - center) * scale
    return vertices, np.asarray(mesh.faces, dtype=np.int64)


def load_dynamic_model(model_dir: Path, checkpoint_path: Path, frame_count: int) -> DynamicGaussianModel:
    dataset = SimpleNamespace(model_path=str(model_dir), sh_degree=3)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)
    model = DynamicGaussianModel(
        gaussians,
        frame_count,
        scene.cameras_extent,
    )
    model.load_pth(str(checkpoint_path))
    return model


@torch.no_grad()
def deform_trajectory(model: DynamicGaussianModel, vertices: np.ndarray, frame_count: int) -> np.ndarray:
    vertices_cuda = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
    frames = []
    for frame in range(frame_count):
        frames.append(model.query_xyz_time(vertices_cuda, frame).cpu().numpy())
    return np.stack(frames, axis=0)


def write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P0 trajectory extraction requires CUDA because CHORD checkpoints are CUDA models.")

    scene_path = args.scene_dir / "scene.glb"
    cat_vertices, cat_faces = load_normalized_mesh(args.scene_dir / "obj_0.glb", scene_path)
    cushion_vertices, cushion_faces = load_normalized_mesh(args.scene_dir / "obj_2.glb", scene_path)
    checkpoint_dir = args.model_dir / args.experiment / "deform" / f"deform_{args.checkpoint}"

    cat_model = load_dynamic_model(args.model_dir / "obj_0", checkpoint_dir / "obj_0.pth", args.frame_count)
    cushion_model = load_dynamic_model(
        args.model_dir / "obj_2", checkpoint_dir / "obj_2.pth", args.frame_count
    )
    cat_trajectory = deform_trajectory(cat_model, cat_vertices, args.frame_count)
    cushion_trajectory = deform_trajectory(cushion_model, cushion_vertices, args.frame_count)

    foot_indices = select_low_vertices(cat_vertices, args.foot_quantile)
    foot_component_count, foot_component_labels = label_vertex_components(
        cat_faces, foot_indices, len(cat_vertices)
    )
    foot_component_sizes = np.bincount(foot_component_labels)
    surface_template = sample_surface_template(
        cushion_vertices,
        cushion_faces,
        args.surface_samples,
        args.seed,
    )
    cushion_diagonal = float(np.linalg.norm(np.ptp(cushion_vertices, axis=0)))
    threshold = args.contact_threshold
    if threshold is None:
        threshold = cushion_diagonal * 0.01

    rows: list[dict[str, float | int]] = []
    for frame in range(args.frame_count):
        surface_points, surface_normals = deform_surface_samples(
            cushion_trajectory[frame], cushion_faces, surface_template
        )
        metrics = frame_contact_metrics(
            cat_trajectory[frame],
            foot_indices,
            surface_points,
            surface_normals,
            threshold,
        )
        cat_delta = cat_trajectory[frame] - cat_trajectory[0]
        cushion_delta = cushion_trajectory[frame] - cushion_trajectory[0]
        cat_step = (
            np.zeros_like(cat_delta)
            if frame == 0
            else cat_trajectory[frame] - cat_trajectory[frame - 1]
        )
        cushion_step = (
            np.zeros_like(cushion_delta)
            if frame == 0
            else cushion_trajectory[frame] - cushion_trajectory[frame - 1]
        )
        rows.append(
            {
                "frame": frame,
                **metrics,
                "cat_mean_displacement_from_frame0": float(np.linalg.norm(cat_delta, axis=1).mean()),
                "cat_max_displacement_from_frame0": float(np.linalg.norm(cat_delta, axis=1).max()),
                "cushion_mean_displacement_from_frame0": float(
                    np.linalg.norm(cushion_delta, axis=1).mean()
                ),
                "cushion_max_displacement_from_frame0": float(
                    np.linalg.norm(cushion_delta, axis=1).max()
                ),
                "cat_mean_frame_step": float(np.linalg.norm(cat_step, axis=1).mean()),
                "cushion_mean_frame_step": float(np.linalg.norm(cushion_step, axis=1).mean()),
            }
        )

    near_counts = np.asarray([row["near_candidate_count"] for row in rows])
    raw_windows = infer_contact_windows(
        near_counts,
        padding=0,
        min_count=args.min_contact_candidates,
    )
    padded_windows = infer_contact_windows(
        near_counts,
        padding=args.window_padding,
        min_count=args.min_contact_candidates,
    )
    output_dir = args.output_dir or (
        args.model_dir / args.experiment / "contact_diagnostics" / f"deform_{args.checkpoint}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "trajectories.npz",
        cat_vertices=cat_trajectory,
        cat_faces=cat_faces,
        cushion_vertices=cushion_trajectory,
        cushion_faces=cushion_faces,
        foot_candidate_indices=foot_indices,
        foot_component_labels=foot_component_labels,
        surface_face_indices=surface_template.face_indices,
        surface_barycentric=surface_template.barycentric,
    )
    write_metrics(output_dir / "frame_metrics.csv", rows)

    min_candidate_frame = int(np.argmin([row["candidate_min_distance"] for row in rows]))
    summary = {
        "scene": str(args.scene_dir),
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "frame_count": args.frame_count,
        "cat_vertex_count": int(len(cat_vertices)),
        "cushion_vertex_count": int(len(cushion_vertices)),
        "foot_candidate_count": int(len(foot_indices)),
        "foot_component_count": int(foot_component_count),
        "foot_component_sizes": sorted(
            (int(value) for value in foot_component_sizes), reverse=True
        ),
        "foot_quantile": args.foot_quantile,
        "surface_sample_count": args.surface_samples,
        "contact_threshold": threshold,
        "contact_threshold_relative_to_cushion_diagonal": threshold / cushion_diagonal,
        "minimum_contact_candidates_per_frame": args.min_contact_candidates,
        "raw_near_contact_runs_inclusive": [list(window) for window in raw_windows],
        "padded_contact_windows_inclusive": [list(window) for window in padded_windows],
        "near_contact_frames": np.flatnonzero(
            near_counts >= args.min_contact_candidates
        ).tolist(),
        "closest_candidate_frame": min_candidate_frame,
        "closest_candidate_distance": rows[min_candidate_frame]["candidate_min_distance"],
        "minimum_all_vertex_distance": min(row["all_min_distance"] for row in rows),
        "signed_distance_warning": (
            "obj_2.glb is non-watertight; oriented gaps are proxies, not exact penetration depths."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[P0] Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
