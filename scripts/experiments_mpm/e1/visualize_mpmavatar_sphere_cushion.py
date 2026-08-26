#!/usr/bin/env python3
"""Render the MPMAvatar sphere/cushion trajectory as a three-view MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("trained/cat_with_cushion/mpmavatar_e1_smoke"),
    )
    parser.add_argument("--output-name", default="sphere_cushion_3d.mp4")
    parser.add_argument("--sphere-radius", type=float, default=0.075)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=110)
    return parser.parse_args()


def penetration_depths(
    particles: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
) -> np.ndarray:
    return np.maximum(
        sphere_radius - np.linalg.norm(particles - sphere_center[None], axis=1),
        0.0,
    )


def display_coordinates(points: np.ndarray) -> np.ndarray:
    """Convert simulation X/Y-up/Z to plotting X/Z/Y-up."""
    return np.asarray(points)[..., [0, 2, 1]]


def equal_bounds(particles: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    lower = np.minimum(particles.min(axis=(0, 1)), centers.min(axis=0) - radius)
    upper = np.maximum(particles.max(axis=(0, 1)), centers.max(axis=0) + radius)
    return display_coordinates(np.stack([lower, upper]))


def set_bounds(axis, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = 0.53 * np.ptp(bounds, axis=0).max()
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def render_video(
    path: Path,
    particles: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radius: float,
    fps: int,
    dpi: int,
) -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
    sphere_vertices = np.asarray(sphere.vertices, dtype=np.float32)
    sphere_faces = np.asarray(sphere.faces, dtype=np.int64)
    bounds = equal_bounds(particles, sphere_centers, sphere_radius)
    cameras = {
        "perspective": (20, -65),
        "front": (0, -90),
        "top": (90, -90),
    }
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame in range(len(particles)):
            depths = penetration_depths(particles[frame], sphere_centers[frame], sphere_radius)
            inside = depths > 0.0
            display_particles = display_coordinates(particles[frame])
            display_sphere = display_coordinates(sphere_vertices + sphere_centers[frame])
            figure = plt.figure(figsize=(15, 5.2), dpi=dpi, constrained_layout=True)
            for index, (name, camera) in enumerate(cameras.items(), start=1):
                axis = figure.add_subplot(1, 3, index, projection="3d")
                axis.scatter(
                    display_particles[~inside, 0], display_particles[~inside, 1],
                    display_particles[~inside, 2], s=3, c="#4c9f70", alpha=0.55,
                    depthshade=False,
                )
                if inside.any():
                    axis.scatter(
                        display_particles[inside, 0], display_particles[inside, 1],
                        display_particles[inside, 2], s=14, c="crimson", depthshade=False,
                    )
                sphere_mesh = Poly3DCollection(
                    display_sphere[sphere_faces], facecolor="#4f83cc", edgecolor="#24527a",
                    alpha=0.24, linewidth=0.25,
                )
                axis.add_collection3d(sphere_mesh)
                set_bounds(axis, bounds)
                axis.view_init(*camera)
                axis.set_title(name)
            figure.suptitle(
                f"MPMAvatar sphere–cushion · frame {frame:02d}/{len(particles)-1:02d} · "
                f"inside particles {int(inside.sum())} · max penetration {depths.max():.4f}"
            )
            figure.canvas.draw()
            writer.append_data(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
            plt.close(figure)


def main() -> None:
    args = parse_args()
    archive = np.load(args.input_dir / "trajectory.npz")
    particles = archive["particles"]
    centers = archive["sphere_centers"]
    output = args.input_dir / args.output_name
    render_video(output, particles, centers, args.sphere_radius, args.fps, args.dpi)
    print(f"[MPMAvatar E1 vis] Wrote {output}")


if __name__ == "__main__":
    main()
