#!/usr/bin/env python3
"""Visualize P0 cat/cushion contact diagnostics from saved trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "trained/cat_with_cushion/cat_with_cushion_izar_v100/"
            "contact_diagnostics/deform_3000"
        ),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--video-name", default="contact_trajectory.mp4")
    parser.add_argument("--video-dpi", type=int, default=120)
    parser.add_argument("--max-cat-faces", type=int, default=6000)
    parser.add_argument("--max-cushion-faces", type=int, default=9000)
    parser.add_argument("--trail-length", type=int, default=8)
    return parser.parse_args()


def load_metric_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def contact_mask(
    foot_vertices: np.ndarray,
    cushion_vertices: np.ndarray,
    threshold: float,
) -> np.ndarray:
    distances, _ = cKDTree(cushion_vertices).query(foot_vertices, k=1)
    return distances <= threshold


def component_centers(
    cat_trajectory: np.ndarray,
    foot_indices: np.ndarray,
    component_labels: np.ndarray,
) -> np.ndarray:
    component_count = int(component_labels.max()) + 1
    return np.stack(
        [
            cat_trajectory[:, foot_indices[component_labels == component], :].mean(axis=1)
            for component in range(component_count)
        ],
        axis=1,
    )


def sampled_faces(faces: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(faces) <= maximum:
        return faces
    indices = np.linspace(0, len(faces) - 1, maximum, dtype=np.int64)
    return faces[indices]


def add_contact_windows(axis: plt.Axes, windows: list[list[int]]) -> None:
    for start, end in windows:
        axis.axvspan(start, end, color="#ffbf69", alpha=0.18, linewidth=0)


def save_statistics(
    path: Path,
    metrics: dict[str, np.ndarray],
    summary: dict,
) -> None:
    frames = metrics["frame"]
    windows = summary["padded_contact_windows_inclusive"]
    threshold = float(summary["contact_threshold"])
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    axes[0].plot(frames, metrics["candidate_min_distance"], label="foot candidate min", lw=2)
    axes[0].plot(frames, metrics["all_min_distance"], label="all cat vertices min", lw=1.3)
    axes[0].axhline(threshold, color="crimson", ls="--", label=f"threshold={threshold:.4f}")
    axes[0].set_ylabel("distance")
    axes[0].set_yscale("symlog", linthresh=max(threshold * 0.25, 1.0e-5))
    axes[0].legend(loc="upper right")

    axes[1].plot(frames, metrics["near_candidate_count"], color="#d1495b", lw=2)
    axes[1].fill_between(frames, 0, metrics["near_candidate_count"], color="#d1495b", alpha=0.2)
    axes[1].set_ylabel("near foot vertices")

    axes[2].plot(frames, metrics["min_oriented_gap_proxy"], label="min oriented gap proxy")
    axes[2].plot(
        frames,
        -metrics["negative_near_gap_mean_depth_proxy"],
        label="negative mean depth proxy",
    )
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("oriented gap")
    axes[2].set_xlabel("frame")
    axes[2].legend(loc="lower right")
    for axis in axes:
        add_contact_windows(axis, windows)
        axis.grid(alpha=0.2)
    figure.suptitle("P0 cat–cushion contact diagnostics\norange: padded contact windows")
    figure.savefig(path, dpi=180)
    plt.close(figure)


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


def draw_scene(
    axis,
    cat_vertices: np.ndarray,
    cushion_vertices: np.ndarray,
    cat_faces: np.ndarray,
    cushion_faces: np.ndarray,
    foot_indices: np.ndarray,
    component_labels: np.ndarray,
    threshold: float,
    bounds: np.ndarray,
    view: str,
) -> np.ndarray:
    cat_vertices = display_coordinates(cat_vertices)
    cushion_vertices = display_coordinates(cushion_vertices)
    bounds = display_coordinates(bounds)
    cat_mesh = Poly3DCollection(cat_vertices[cat_faces], alpha=0.28, linewidths=0.0)
    cat_mesh.set_facecolor("#9b9b9b")
    cushion_mesh = Poly3DCollection(cushion_vertices[cushion_faces], alpha=0.32, linewidths=0.0)
    cushion_mesh.set_facecolor("#65a9d8")
    axis.add_collection3d(cat_mesh)
    axis.add_collection3d(cushion_mesh)
    candidates = cat_vertices[foot_indices]
    colors = plt.get_cmap("tab10")(component_labels % 10)
    axis.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], c=colors, s=8, depthshade=False)
    near = contact_mask(candidates, cushion_vertices, threshold)
    if near.any():
        points = candidates[near]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c="red", s=28, marker="o", depthshade=False)
    set_equal_bounds(axis, bounds)
    camera = {
        "front": (0, -90),   # look along original Z; show X–Y
        "side": (0, 0),      # look along original X; show Z–Y
        "top": (90, -90),    # look along original Y; show X–Z
    }
    axis.view_init(*camera[view])
    axis.set_axis_off()
    return near


def save_candidate_overview(path: Path, data, summary: dict, cat_faces, cushion_faces) -> None:
    closest = int(summary["closest_candidate_frame"])
    frames = [0, closest]
    bounds = np.stack(
        [
            np.minimum(data["cat_vertices"].min(axis=(0, 1)), data["cushion_vertices"].min(axis=(0, 1))),
            np.maximum(data["cat_vertices"].max(axis=(0, 1)), data["cushion_vertices"].max(axis=(0, 1))),
        ]
    )
    views = ("front", "side", "top")
    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    for row, frame in enumerate(frames):
        for column, view in enumerate(views):
            axis = figure.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            near = draw_scene(
                axis, data["cat_vertices"][frame], data["cushion_vertices"][frame],
                cat_faces, cushion_faces, data["foot_candidate_indices"],
                data["foot_component_labels"], float(summary["contact_threshold"]), bounds,
                view,
            )
            axis.set_title(
                f"frame {frame} · {view}\n{int(near.sum())} vertex-nearest contacts"
            )
    figure.suptitle("Foot candidate components (color) and near-contact candidates (red)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_video(path: Path, data, summary: dict, args, cat_faces, cushion_faces) -> None:
    cat = data["cat_vertices"]
    cushion = data["cushion_vertices"]
    foot_indices = data["foot_candidate_indices"]
    labels = data["foot_component_labels"]
    centers = component_centers(cat, foot_indices, labels)
    bounds = np.stack(
        [np.minimum(cat.min(axis=(0, 1)), cushion.min(axis=(0, 1))),
         np.maximum(cat.max(axis=(0, 1)), cushion.max(axis=(0, 1)))],
    )
    threshold = float(summary["contact_threshold"])
    with imageio.get_writer(path, fps=args.fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for frame in range(len(cat)):
            figure = plt.figure(figsize=(15, 5.2), dpi=args.video_dpi, constrained_layout=True)
            near = None
            start = max(0, frame - args.trail_length + 1)
            for view_index, view in enumerate(("front", "side", "top"), start=1):
                axis = figure.add_subplot(1, 3, view_index, projection="3d")
                near = draw_scene(
                    axis, cat[frame], cushion[frame], cat_faces, cushion_faces,
                    foot_indices, labels, threshold, bounds, view,
                )
                for component in range(centers.shape[1]):
                    trail = display_coordinates(centers[start : frame + 1, component])
                    axis.plot(trail[:, 0], trail[:, 1], trail[:, 2], lw=2,
                              color=plt.get_cmap("tab10")(component % 10))
                axis.set_title(view)
            figure.suptitle(
                f"P0 deformed mesh trajectory — frame {frame:02d}/{len(cat)-1:02d} · "
                f"red near-contact candidates: {int(near.sum())}"
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = load_metric_rows(input_dir / "frame_metrics.csv")
    archive = np.load(input_dir / "trajectories.npz")
    data = {key: archive[key] for key in archive.files}
    cat_faces = sampled_faces(data["cat_faces"], args.max_cat_faces)
    cushion_faces = sampled_faces(data["cushion_faces"], args.max_cushion_faces)

    save_statistics(input_dir / "contact_statistics.png", metrics, summary)
    save_candidate_overview(
        input_dir / "foot_candidates.png", data, summary, cat_faces, cushion_faces
    )
    render_video(input_dir / args.video_name, data, summary, args, cat_faces, cushion_faces)
    print(f"[P0-vis] Wrote visualizations to {input_dir}")


if __name__ == "__main__":
    main()
