#!/usr/bin/env python3
"""E2 step 1: export and visualize the CHORD-trained active cat motion.

Loads the obj_0 deformation checkpoint, evaluates the deformed cat mesh in the
CHORD normalized scene space for every frame, saves the trajectory as npz for
the MPM collider driver (exp16_chord_motion.py in MPMAvatar), and renders a
visualization video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import GaussianModel, Scene
from scene.dynamic_gaussian_model import DynamicGaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--model-dir", type=Path, default=Path("trained/cat_with_cushion"))
    parser.add_argument("--experiment", default="cat_with_cushion_izar_v100")
    parser.add_argument("--checkpoint", type=int, default=3000)
    parser.add_argument("--frame-count", type=int, default=41)
    parser.add_argument("--foot-quantile", type=float, default=0.12)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-dpi", type=int, default=120)
    parser.add_argument("--max-cushion-faces", type=int, default=9000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trained/cat_with_cushion/outputs_mpmavatar_e2/chord_cat_motion"),
    )
    return parser.parse_args()


def scene_normalization(scene_path: Path) -> tuple[np.ndarray, float]:
    scene = trimesh.load(scene_path, force="mesh", process=False)
    center = scene.bounds.mean(axis=0)
    scale = 1.2 / np.ptp(scene.bounds, axis=0).max()
    return center.astype(np.float64), float(scale)


def load_normalized_mesh(path: Path, center: np.ndarray, scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) * scale
    return mesh


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


@torch.no_grad()
def control_point_trajectory(model: DynamicGaussianModel, frame_count: int) -> dict[str, np.ndarray]:
    trajectories = {}
    base = np.stack(
        [model.get_cp_position(frame).cpu().numpy() for frame in range(frame_count)],
        axis=0,
    )
    trajectories["cp_base"] = base.astype(np.float32)
    if model.c_cp_deform is not None:
        additional = np.stack(
            [model.get_additional_cp_position(frame).cpu().numpy() for frame in range(frame_count)],
            axis=0,
        )
        trajectories["cp_additional"] = additional.astype(np.float32)
    return trajectories


def display_coordinates(points: np.ndarray) -> np.ndarray:
    """Map simulation (X, Y-up, Z) to plotting (X, Z, Y-up)."""
    return points[..., [0, 2, 1]]


def set_equal_bounds(axis, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = float(np.ptp(bounds, axis=0).max()) * 0.52
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def sampled_faces(faces: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(faces) <= maximum:
        return faces
    indices = np.linspace(0, len(faces) - 1, maximum, dtype=np.int64)
    return faces[indices]


def render_video(
    path: Path,
    cat_trajectory: np.ndarray,
    cat_faces: np.ndarray,
    foot_indices: np.ndarray,
    cushion_vertices: np.ndarray,
    cushion_faces: np.ndarray,
    fps: int,
    dpi: int,
) -> None:
    bounds = np.stack(
        [
            np.minimum(cat_trajectory.min(axis=(0, 1)), cushion_vertices.min(axis=0)),
            np.maximum(cat_trajectory.max(axis=(0, 1)), cushion_vertices.max(axis=0)),
        ]
    )
    views = {"front": (0, -90), "side": (0, 0), "top": (90, -90)}
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame in range(len(cat_trajectory)):
            figure = plt.figure(figsize=(15, 5.2), dpi=dpi, constrained_layout=True)
            cat = display_coordinates(cat_trajectory[frame])
            for view_index, (view_name, view_angle) in enumerate(views.items(), start=1):
                axis = figure.add_subplot(1, 3, view_index, projection="3d")
                cushion_mesh = Poly3DCollection(
                    display_coordinates(cushion_vertices)[cushion_faces],
                    alpha=0.20,
                    linewidths=0.0,
                )
                cushion_mesh.set_facecolor("#65a9d8")
                cat_mesh = Poly3DCollection(cat[cat_faces], alpha=0.45, linewidths=0.0)
                cat_mesh.set_facecolor("#d1495b")
                axis.add_collection3d(cushion_mesh)
                axis.add_collection3d(cat_mesh)
                foot = cat[foot_indices]
                axis.scatter(foot[:, 0], foot[:, 1], foot[:, 2], c="#2b8c67", s=6, depthshade=False)
                set_equal_bounds(axis, display_coordinates(bounds))
                axis.view_init(*view_angle)
                axis.set_axis_off()
                axis.set_title(view_name)
            figure.suptitle(
                f"CHORD-trained active cat motion — frame {frame:02d}/{len(cat_trajectory)-1:02d}"
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CHORD checkpoints are CUDA models; run on a GPU node.")

    center, scale = scene_normalization(args.scene_dir / "scene.glb")
    cat_mesh = load_normalized_mesh(args.scene_dir / "obj_0.glb", center, scale)
    cushion_mesh = load_normalized_mesh(args.scene_dir / "obj_2.glb", center, scale)
    cat_vertices = np.asarray(cat_mesh.vertices, dtype=np.float32)
    cat_faces = np.asarray(cat_mesh.faces, dtype=np.int64)
    cushion_vertices = np.asarray(cushion_mesh.vertices, dtype=np.float32)
    cushion_faces = np.asarray(cushion_mesh.faces, dtype=np.int64)

    checkpoint_path = (
        args.model_dir
        / args.experiment
        / "deform"
        / f"deform_{args.checkpoint}"
        / "obj_0.pth"
    )
    model = load_dynamic_model(
        args.model_dir / "obj_0", checkpoint_path, args.frame_count
    )
    cat_trajectory = deform_trajectory(model, cat_vertices, args.frame_count)
    cp_trajectories = control_point_trajectory(model, args.frame_count)

    foot_indices = np.flatnonzero(
        cat_vertices[:, 1] <= np.quantile(cat_vertices[:, 1], args.foot_quantile)
    )
    foot_centers = cat_trajectory[:, foot_indices, :].mean(axis=1)

    displacement = cat_trajectory - cat_trajectory[0]
    displacement_norm = np.linalg.norm(displacement, axis=2)
    stats = {
        "scene_dir": str(args.scene_dir),
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "frame_count": args.frame_count,
        "fps": args.fps,
        "note": "frames beyond the active window are clamped to the last active frame by assign_deform",
        "normalization": {
            "center": center.tolist(),
            "scale": scale,
            "formula": "v_norm = (v_glb - center) * scale",
        },
        "cat_vertex_count": int(len(cat_vertices)),
        "cat_face_count": int(len(cat_faces)),
        "foot_quantile": args.foot_quantile,
        "foot_vertex_count": int(len(foot_indices)),
        "per_frame_mean_displacement": displacement_norm.mean(axis=1).tolist(),
        "per_frame_max_displacement": displacement_norm.max(axis=1).tolist(),
        "foot_center": foot_centers.tolist(),
        "motion_frame_range": [
            int(np.flatnonzero(displacement_norm.max(axis=1) > 0)[0]),
            int(np.flatnonzero(displacement_norm.max(axis=1) > 0)[-1]),
        ],
        "trajectory_finite": bool(np.isfinite(cat_trajectory).all()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "chord_cat_motion.npz",
        vertices=cat_trajectory.astype(np.float32),
        faces=cat_faces.astype(np.int64),
        foot_indices=foot_indices.astype(np.int64),
        foot_center=foot_centers.astype(np.float32),
        frame_count=np.asarray([args.frame_count], dtype=np.int64),
        fps=np.asarray([args.fps], dtype=np.int64),
        center=center.astype(np.float32),
        scale=np.asarray([scale], dtype=np.float64),
        **cp_trajectories,
    )
    with (args.output_dir / "motion_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    render_video(
        args.output_dir / "chord_cat_motion.mp4",
        cat_trajectory,
        cat_faces,
        foot_indices,
        cushion_vertices,
        sampled_faces(cushion_faces, args.max_cushion_faces),
        args.fps,
        args.video_dpi,
    )
    print(json.dumps(stats, indent=2))
    print(f"[E2] Wrote cat motion export to {args.output_dir}")


if __name__ == "__main__":
    main()
