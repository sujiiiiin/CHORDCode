#!/usr/bin/env python3
"""Render force annotation plus particle and reconstructed-mesh three-view videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments_mpm.e1.cushion_volume_particles import load_chord_normalized_mesh

CAMERAS = {"perspective": (20, -65), "front": (0, -90), "top": (90, -90)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("trained/cat_with_cushion/mpmavatar_e1_impulse"),
    )
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--mesh-voxel-size", type=float, default=0.0,
        help="Display-only vertex clustering size; 0 keeps the original mesh.",
    )
    return parser.parse_args()


def display(points: np.ndarray) -> np.ndarray:
    return np.asarray(points)[..., [0, 2, 1]]


def set_view(axis, bounds: np.ndarray, camera: tuple[int, int]) -> None:
    center = bounds.mean(axis=0)
    radius = 0.55 * np.ptp(bounds, axis=0).max()
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(*camera)
    axis.set_axis_off()


def force_arrow(axis, point: np.ndarray, direction: np.ndarray, scale: float) -> None:
    p, d = display(point), display(direction)
    axis.quiver(*p, *d, length=scale, normalize=True, color="crimson", linewidth=3,
                arrow_length_ratio=0.25)


def render_force_image(path: Path, particles: np.ndarray, mask: np.ndarray,
                       point: np.ndarray, direction: np.ndarray, bounds: np.ndarray) -> None:
    shown = display(particles)
    figure = plt.figure(figsize=(15, 5.2), dpi=120, constrained_layout=True)
    for index, (name, camera) in enumerate(CAMERAS.items(), 1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(*shown[~mask].T, s=3, c="#4c9f70", alpha=0.45)
        axis.scatter(*shown[mask].T, s=18, c="gold", edgecolor="darkorange")
        force_arrow(axis, point, direction, 0.12)
        set_view(axis, bounds, camera)
        axis.set_title(name)
    figure.suptitle("Initial force: yellow = acted-on particles, red arrow = direction")
    figure.savefig(path)
    plt.close(figure)


def render_particles(path: Path, snapshots: np.ndarray, mask: np.ndarray,
                     point: np.ndarray, direction: np.ndarray, bounds: np.ndarray,
                     fps: int) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2) as writer:
        for frame, particles in enumerate(snapshots):
            shown = display(particles)
            figure = plt.figure(figsize=(15, 5.2), dpi=105, constrained_layout=True)
            for index, (name, camera) in enumerate(CAMERAS.items(), 1):
                axis = figure.add_subplot(1, 3, index, projection="3d")
                axis.scatter(*shown[~mask].T, s=3, c="#4c9f70", alpha=0.5)
                axis.scatter(*shown[mask].T, s=12, c="gold")
                if frame == 0:
                    force_arrow(axis, point, direction, 0.12)
                set_view(axis, bounds, camera)
                axis.set_title(name)
            figure.suptitle(f"MPM cushion particles · frame {frame:02d}/{len(snapshots)-1:02d}")
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            plt.close(figure)


def reconstruct_mesh(surface_vertices: np.ndarray, initial_particles: np.ndarray,
                     snapshots: np.ndarray) -> np.ndarray:
    distances, indices = cKDTree(initial_particles).query(surface_vertices, k=8)
    weights = 1.0 / np.maximum(distances, 1.0e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    displacement = snapshots - initial_particles[None]
    return surface_vertices[None] + np.sum(
        displacement[:, indices] * weights[None, :, :, None], axis=2
    )


def cluster_mesh(vertices: np.ndarray, faces: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Simplify only the rendered surface with deterministic vertex clustering."""
    if size <= 0.0:
        return vertices, faces
    cells = np.floor((vertices - vertices.min(axis=0)) / size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    clustered = np.stack([
        np.bincount(inverse, weights=vertices[:, axis]) / counts for axis in range(3)
    ], axis=1).astype(np.float32)
    clustered_faces = inverse[faces]
    keep = (
        (clustered_faces[:, 0] != clustered_faces[:, 1])
        & (clustered_faces[:, 1] != clustered_faces[:, 2])
        & (clustered_faces[:, 0] != clustered_faces[:, 2])
    )
    clustered_faces = np.unique(np.sort(clustered_faces[keep], axis=1), axis=0)
    return clustered, clustered_faces


def render_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray,
                point: np.ndarray, direction: np.ndarray, bounds: np.ndarray,
                fps: int) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=2) as writer:
        for frame, frame_vertices in enumerate(vertices):
            shown = display(frame_vertices)
            figure = plt.figure(figsize=(15, 5.2), dpi=105, constrained_layout=True)
            for index, (name, camera) in enumerate(CAMERAS.items(), 1):
                axis = figure.add_subplot(1, 3, index, projection="3d")
                axis.add_collection3d(Poly3DCollection(
                    shown[faces], facecolor="#58a879", edgecolor="#275b3b",
                    linewidth=0.15, alpha=0.85,
                ))
                if frame == 0:
                    force_arrow(axis, point, direction, 0.12)
                set_view(axis, bounds, camera)
                axis.set_title(name)
            figure.suptitle(
                f"Cushion surface reconstructed from MPM particles · "
                f"frame {frame:02d}/{len(vertices)-1:02d}"
            )
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            plt.close(figure)


def main() -> None:
    args = parse_args()
    archive = np.load(args.input_dir / "trajectory.npz")
    snapshots = archive["particles"]
    mask = archive["force_mask"].astype(bool)
    point, direction = archive["force_point"], archive["force_direction"]
    bounds = display(np.stack([snapshots.min(axis=(0, 1)), snapshots.max(axis=(0, 1))]))
    mesh = load_chord_normalized_mesh(args.scene_dir)
    surface_vertices, surface_faces = cluster_mesh(
        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces),
        args.mesh_voxel_size,
    )
    surface_vertices = surface_vertices + archive["mesh_shift"]
    mesh_trajectory = reconstruct_mesh(surface_vertices, snapshots[0], snapshots)
    render_force_image(args.input_dir / "force_initial.png", snapshots[0], mask,
                       point, direction, bounds)
    render_particles(args.input_dir / "cushion_particles_3d.mp4", snapshots, mask,
                     point, direction, bounds, args.fps)
    render_mesh(args.input_dir / "cushion_mesh_3d.mp4", mesh_trajectory,
                surface_faces, point, direction, bounds, args.fps)
    print(f"[MPMAvatar E1 impulse vis] Wrote outputs to {args.input_dir}")


if __name__ == "__main__":
    main()
