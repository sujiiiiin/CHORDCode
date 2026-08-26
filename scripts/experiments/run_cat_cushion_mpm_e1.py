#!/usr/bin/env python3
"""E1: direct MPM mechanism test with a kinematic sphere and cushion proxy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mpm_utils import collider_state, column_fill_points, lame_parameters
from utils.warp_mpm import MPMConfig, WarpMPMSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained/cat_with_cushion/mpm_e1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proxy-pitch", type=float, default=0.015)
    parser.add_argument("--grid-dx", type=float, default=0.025)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--extra-recovery-duration", type=float, default=0.0)
    parser.add_argument("--output-frames", type=int, default=51)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.9995)
    parser.add_argument("--fixed-bottom-layers", type=int, default=2)
    parser.add_argument(
        "--particle-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--collider-radius", type=float, default=0.075)
    parser.add_argument("--press-depth", type=float, default=0.045)
    parser.add_argument("--slide-distance", type=float, default=0.10)
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=("no_contact", "contact_friction0", "contact_friction04"),
        default=("no_contact", "contact_friction0", "contact_friction04"),
    )
    return parser.parse_args()


def normalized_mesh(scene_dir: Path) -> trimesh.Trimesh:
    scene = trimesh.load(scene_dir / "scene.glb", force="mesh", process=False)
    cushion = trimesh.load(scene_dir / "obj_2.glb", force="mesh", process=False)
    center = scene.bounds.mean(axis=0)
    scale = 1.2 / np.ptp(scene.bounds, axis=0).max()
    cushion.vertices = (cushion.vertices - center) * scale
    return cushion


def build_volume_proxy(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    surface = mesh.voxelized(pitch).points
    # obj_2.glb is open at parts of its underside. Treat the cushion as a
    # height-field solid instead of pretending that generic hole filling worked.
    particles = column_fill_points(surface, pitch, fill_to_global_bottom=True)
    return np.unique(np.round(particles, 7), axis=0).astype(np.float32)


def state_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    deformation: np.ndarray,
    rest: np.ndarray,
    fixed: np.ndarray,
    local_top: np.ndarray,
    far: np.ndarray,
    center: np.ndarray,
    radius: float,
    mu: float,
    lam: float,
) -> dict[str, float]:
    displacement = positions - rest
    distances = np.linalg.norm(positions - center[None], axis=1)
    penetration = np.maximum(radius - distances, 0.0)
    local_down = np.maximum(rest[local_top, 1] - positions[local_top, 1], 0.0)
    movable = fixed == 0
    determinants = np.linalg.det(deformation[movable])
    identity = np.eye(3, dtype=deformation.dtype)
    f_error = np.linalg.norm(deformation[movable] - identity, axis=(1, 2))
    safe_determinants = np.maximum(determinants, 1.0e-8)
    log_j = np.log(safe_determinants)
    first_invariant = np.square(deformation[movable]).sum(axis=(1, 2))
    elastic_energy_density = (
        0.5 * mu * (first_invariant - 3.0)
        - mu * log_j
        + 0.5 * lam * np.square(log_j)
    )
    return {
        "max_sphere_penetration": float(penetration.max()),
        "local_mean_downward_displacement": float(local_down.mean()),
        "local_max_downward_displacement": float(local_down.max()),
        "local_mean_x_displacement": float(displacement[local_top, 0].mean()),
        "far_mean_displacement": float(np.linalg.norm(displacement[far], axis=1).mean()),
        "movable_rmse_from_rest": float(np.sqrt(np.square(displacement[movable]).mean())),
        "mean_abs_volume_change": float(np.abs(determinants - 1.0).mean()),
        "min_deformation_determinant": float(determinants.min()),
        "max_deformation_determinant": float(determinants.max()),
        "mean_frobenius_F_minus_I": float(f_error.mean()),
        "max_frobenius_F_minus_I": float(f_error.max()),
        "mean_elastic_energy_density": float(elastic_energy_density.mean()),
        "max_elastic_energy_density": float(elastic_energy_density.max()),
        "mean_speed": float(np.linalg.norm(velocities[movable], axis=1).mean()),
        "kinetic_energy_proxy": float(0.5 * np.square(velocities[movable]).sum()),
    }


def run_control(
    name: str,
    particles: np.ndarray,
    fixed: np.ndarray,
    config: MPMConfig,
    args: argparse.Namespace,
    collider_enabled: bool,
    friction: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    solver = WarpMPMSolver(particles, fixed, config, args.device)
    radius = args.collider_radius
    top_y = float(particles[:, 1].max())
    start_center = np.array(
        [np.median(particles[:, 0]) - args.slide_distance * 0.5, top_y + radius + 0.02, 0.0],
        dtype=np.float32,
    )
    top = particles[:, 1] >= np.quantile(particles[:, 1], 0.80)
    path_x_min = start_center[0] - radius
    path_x_max = start_center[0] + args.slide_distance + radius
    local_top = top & (particles[:, 0] >= path_x_min) & (particles[:, 0] <= path_x_max)
    local_top &= np.abs(particles[:, 2] - start_center[2]) <= radius
    far = np.abs(particles[:, 2] - start_center[2]) >= 2.5 * radius
    far &= fixed == 0
    if not local_top.any() or not far.any():
        raise RuntimeError("Failed to construct local/far particle masks")

    total_duration = args.duration + args.extra_recovery_duration
    if total_duration <= 0.0:
        raise ValueError("duration plus extra recovery duration must be positive")
    substeps = int(round(total_duration / args.dt))
    output_steps = np.linspace(0, substeps, args.output_frames, dtype=np.int64)
    output_lookup = {int(step): index for index, step in enumerate(output_steps)}
    snapshots = np.empty((args.output_frames, len(particles), 3), dtype=np.float32)
    centers = np.empty((args.output_frames, 3), dtype=np.float32)
    rows: list[dict[str, float]] = []
    start_time = time.monotonic()

    for step in range(substeps + 1):
        current_time = step * args.dt
        # Keep the original contact motion unchanged. During the extra interval
        # the collider remains at its lifted final pose with zero velocity.
        collider = collider_state(
            min(current_time, args.duration),
            start_center,
            args.press_depth,
            args.slide_distance,
            args.duration,
        )
        if step in output_lookup:
            wp.synchronize_device(args.device)
            positions = solver.positions()
            velocities = solver.velocities()
            deformation = solver.deformation_gradients()
            output_index = output_lookup[step]
            snapshots[output_index] = positions
            centers[output_index] = collider.center
            rows.append(
                {
                    "control": name,
                    "output_index": output_index,
                    "step": step,
                    "time": current_time,
                    **state_metrics(
                        positions,
                        velocities,
                        deformation,
                        particles,
                        fixed,
                        local_top,
                        far,
                        collider.center,
                        radius,
                        config.mu,
                        config.lam,
                    ),
                }
            )
        if step == substeps:
            break
        solver.step(
            collider.center,
            collider.velocity,
            radius,
            friction,
            collider_enabled,
            args.particle_projection,
        )
    wp.synchronize_device(args.device)
    rows[-1]["wall_time_seconds"] = time.monotonic() - start_time
    return rows, snapshots, centers


def save_visualization(
    path: Path,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    radius: float,
    active_duration: float,
    total_duration: float,
) -> None:
    frame_count = next(iter(results.values()))[0].shape[0]
    selected_times = [0.0, 0.4 * active_duration, 0.6 * active_duration, active_duration]
    if total_duration > active_duration:
        selected_times.extend([(active_duration + total_duration) * 0.5, total_duration])
    selected = np.unique(
        np.rint(np.asarray(selected_times) / total_duration * (frame_count - 1)).astype(int)
    ).tolist()
    figure, axes = plt.subplots(len(results), len(selected), figsize=(15, 8), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(results), len(selected))
    for row, (name, (positions, centers)) in enumerate(results.items()):
        for col, frame in enumerate(selected):
            axis = axes[row, col]
            points = positions[frame]
            keep = np.abs(points[:, 2]) < 0.02
            axis.scatter(points[keep, 0], points[keep, 1], s=2, c=points[keep, 1], cmap="viridis")
            circle = plt.Circle((centers[frame, 0], centers[frame, 1]), radius, fill=False, color="red")
            axis.add_patch(circle)
            frame_time = frame / (frame_count - 1) * total_duration
            axis.set_title(f"{name} t={frame_time:.2f}")
            axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = normalized_mesh(args.scene_dir)
    particles = build_volume_proxy(mesh, args.proxy_pitch)
    bottom = float(particles[:, 1].min())
    if args.fixed_bottom_layers <= 0:
        raise ValueError("fixed-bottom-layers must be positive")
    fixed_threshold = bottom + (args.fixed_bottom_layers - 1 + 0.1) * args.proxy_pitch
    fixed = (particles[:, 1] <= fixed_threshold).astype(np.int32)
    mu, lam = lame_parameters(args.youngs_modulus, args.poisson_ratio)
    config = MPMConfig(
        dx=args.grid_dx,
        dt=args.dt,
        density=args.density,
        particle_volume=args.proxy_pitch**3,
        mu=mu,
        lam=lam,
        damping=args.damping,
    )
    available_controls = {
        "no_contact": (False, 0.0),
        "contact_friction0": (True, 0.0),
        "contact_friction04": (True, 0.4),
    }
    controls = {name: available_controls[name] for name in args.controls}
    all_rows: list[dict[str, float]] = []
    visual_results = {}
    summary = {
        "particle_count": int(len(particles)),
        "fixed_particle_count": int(fixed.sum()),
        "mesh_watertight": bool(mesh.is_watertight),
        "config": vars(args) | {"mu": mu, "lambda": lam},
        "controls": {},
    }
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["config"].items()}

    for name, (enabled, friction) in controls.items():
        rows, snapshots, centers = run_control(
            name, particles, fixed, config, args, enabled, friction
        )
        all_rows.extend(rows)
        visual_results[name] = (snapshots, centers)
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            positions=snapshots,
            collider_centers=centers,
            rest_positions=particles,
            fixed=fixed,
        )
        summary["controls"][name] = {
            "stable": bool(np.isfinite(snapshots).all()),
            "peak_local_mean_indentation": max(row["local_mean_downward_displacement"] for row in rows),
            "peak_local_max_indentation": max(row["local_max_downward_displacement"] for row in rows),
            "peak_penetration": max(row["max_sphere_penetration"] for row in rows),
            "final_recovery_rmse": rows[-1]["movable_rmse_from_rest"],
            "final_local_indentation": rows[-1]["local_mean_downward_displacement"],
            "final_mean_frobenius_F_minus_I": rows[-1]["mean_frobenius_F_minus_I"],
            "final_mean_speed": rows[-1]["mean_speed"],
            "final_mean_elastic_energy_density": rows[-1]["mean_elastic_energy_density"],
            "final_local_x_displacement": rows[-1]["local_mean_x_displacement"],
            "max_mean_abs_volume_change": max(row["mean_abs_volume_change"] for row in rows),
            "wall_time_seconds": rows[-1]["wall_time_seconds"],
        }

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in all_rows for key in row}))
        writer.writeheader()
        writer.writerows(all_rows)
    save_visualization(
        args.output_dir / "side_view.png",
        visual_results,
        args.collider_radius,
        args.duration,
        args.duration + args.extra_recovery_duration,
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[E1] Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
