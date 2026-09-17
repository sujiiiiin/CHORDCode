#!/usr/bin/env python3
"""E1 smoke test: press a rigid cat mesh into an MPM cushion, then release it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments_mpm.e1.cushion_volume_particles import (
    heightfield_volume_particles,
    load_chord_normalized_mesh,
    shift_into_mpm_domain,
)
from scripts.experiments_mpm.e1.run_mpmavatar_sphere_cushion_smoke import (
    initialize_traditional_state,
)
from warp_mpm.mpm_data_structure import MPMModelStruct
from warp_mpm.mpm_solver import MPMWARP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("trained/cat_with_cushion/mpmavatar_e1_cat_cushion_smoke"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pitch", type=float, default=0.02)
    parser.add_argument("--grid-resolution", type=int, default=80)
    parser.add_argument("--grid-limit", type=float, default=1.6)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--duration", type=float, default=7.70)
    parser.add_argument("--output-frames", type=int, default=81)
    parser.add_argument("--initial-gap", type=float, default=0.01)
    parser.add_argument("--press-travel", type=float, default=0.12)
    parser.add_argument("--press-duration", type=float, default=4.80)
    parser.add_argument("--hold-duration", type=float, default=0.20)
    parser.add_argument("--release-duration", type=float, default=2.40)
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--material", choices=("elastic", "plastic"), default="elastic")
    parser.add_argument("--yield-stress", type=float, default=20.0)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    return parser.parse_args()


def collider_motion(
    time: float,
    travel: float,
    press_duration: float,
    hold_duration: float,
    release_duration: float,
) -> tuple[float, float, str]:
    """Return vertical offset, velocity, and phase for press-hold-release motion."""
    if time < press_duration:
        return -travel * time / press_duration, -travel / press_duration, "press"
    release_start = press_duration + hold_duration
    if time < release_start:
        return -travel, 0.0, "hold"
    release_end = release_start + release_duration
    if time < release_end:
        progress = (time - release_start) / release_duration
        return -travel * (1.0 - progress), travel / release_duration, "release"
    return 0.0, 0.0, "free"


def save_cross_section(
    path: Path,
    snapshots: np.ndarray,
    cat_vertices0: np.ndarray,
    cat_faces: np.ndarray,
    cat_offsets: np.ndarray,
) -> None:
    lowest = np.flatnonzero(np.isclose(cat_offsets, cat_offsets.min(), atol=1.0e-6))
    selected = np.array([0, int(lowest[0]), int(lowest[-1]), len(snapshots) - 1])
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
    z_mid = float(np.median(snapshots[0, :, 2]))
    for axis, frame in zip(axes, selected):
        points = snapshots[frame]
        keep = np.abs(points[:, 2] - z_mid) <= 0.015
        cat = cat_vertices0 + np.array([0.0, cat_offsets[frame], 0.0], dtype=np.float32)
        axis.scatter(points[keep, 0], points[keep, 1], s=3, color="#4c9f70")
        segments = []
        for triangle in cat[cat_faces]:
            intersections = []
            for start, end in ((0, 1), (1, 2), (2, 0)):
                z0, z1 = triangle[start, 2] - z_mid, triangle[end, 2] - z_mid
                if z0 * z1 <= 0.0 and z0 != z1:
                    alpha = -z0 / (z1 - z0)
                    intersections.append(triangle[start, :2] + alpha * (triangle[end, :2] - triangle[start, :2]))
            if len(intersections) >= 2:
                segments.append(np.stack(intersections[:2]))
        axis.add_collection(LineCollection(segments, colors="crimson", linewidths=1.0))
        axis.set_title(f"frame {frame}")
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_response_curve(
    path: Path,
    times: np.ndarray,
    local_mean_down: np.ndarray,
    local_max_down: np.ndarray,
    cat_offsets: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(times, local_mean_down, label="local mean", color="#2b8c67")
    axis.plot(times, local_max_down, label="local max", color="#74c476")
    axis.set_xlabel("time")
    axis.set_ylabel("downward cushion displacement")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    motion_axis = axis.twinx()
    motion_axis.plot(times, -cat_offsets, label="cat travel", color="crimson", alpha=0.65)
    motion_axis.set_ylabel("downward cat travel", color="crimson")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    motion_end = args.press_duration + args.hold_duration + args.release_duration
    if min(args.press_duration, args.release_duration) <= 0.0:
        raise ValueError("press-duration and release-duration must be positive")
    if args.duration <= motion_end:
        raise ValueError("duration must leave time after release to observe rebound")

    wp.init()
    cushion_mesh = load_chord_normalized_mesh(args.scene_dir, "obj_2.glb")
    cat_mesh = load_chord_normalized_mesh(args.scene_dir, "obj_0.glb")
    volume = heightfield_volume_particles(cushion_mesh, args.pitch)
    dx = args.grid_limit / args.grid_resolution
    positions, shift = shift_into_mpm_domain(volume.positions, args.grid_limit, 4.0 * dx)

    cat_vertices0 = np.asarray(cat_mesh.vertices, dtype=np.float32) + shift
    cat_vertices0[:, [1, 2]] *= -1.0  # Rotate the rigid cat 180 degrees about X.
    alignment = np.zeros(3, dtype=np.float32)
    alignment[[0, 2]] = (
        0.5 * (positions[:, [0, 2]].min(axis=0) + positions[:, [0, 2]].max(axis=0))
        - 0.5 * (cat_vertices0[:, [0, 2]].min(axis=0) + cat_vertices0[:, [0, 2]].max(axis=0))
    )
    alignment[1] = (
        float(positions[:, 1].max()) + args.initial_gap - float(cat_vertices0[:, 1].min())
    )
    cat_vertices0 += alignment
    cat_faces = np.asarray(cat_mesh.faces, dtype=np.int32)
    if np.any(cat_vertices0.min(axis=0) < 0.0) or np.any(cat_vertices0.max(axis=0) >= args.grid_limit):
        raise ValueError("cat mesh does not fit in the MPM domain; increase --grid-limit")

    state = initialize_traditional_state(
        positions, volume.volumes, args.grid_resolution, args.grid_limit, args.device
    )
    model = MPMModelStruct()
    model.init(len(positions), device=args.device, requires_grad=False)
    model.init_other_params(args.grid_resolution, args.grid_limit, device=args.device)
    solver = MPMWARP(
        len(positions), 0, 0,
        n_grid=args.grid_resolution,
        grid_lim=args.grid_limit,
        mesh_vertices=cat_vertices0,
        mesh_faces=cat_faces,
        device=args.device,
    )
    solver.set_parameters_dict(
        model,
        state,
        {
            "material": "jelly" if args.material == "elastic" else "plasticine",
            "yield_stress": args.yield_stress,
            "g": [0.0, 0.0, 0.0],
            "density": args.density,
            "grid_v_damping_scale": 1.0,
            "rpic_damping": 0.0,
        },
        device=args.device,
    )
    solver.set_E_nu(
        model, args.youngs_modulus, args.poisson_ratio, 0.0, 0.0, device=args.device
    )
    solver.prepare_mu_lam(model, state, device=args.device)
    solver.add_surface_collider(
        [0.0, float(positions[:, 1].min() - 0.5 * args.pitch), 0.0],
        [0.0, 1.0, 0.0],
        surface="sticky",
        friction=0.0,
    )
    solver.add_mesh_collider(
        solver.mesh.id,
        n_grid=args.grid_resolution,
        friction=args.friction,
        device=args.device,
    )

    # Approximate the loaded cat's contact footprint from its lowest vertices.
    low_cat = cat_vertices0[:, 1] <= np.quantile(cat_vertices0[:, 1], 0.12)
    footprint_min = cat_vertices0[low_cat][:, [0, 2]].min(axis=0) - args.pitch
    footprint_max = cat_vertices0[low_cat][:, [0, 2]].max(axis=0) + args.pitch
    top = positions[:, 1] >= np.quantile(positions[:, 1], 0.8)
    local = top & np.all(
        (positions[:, [0, 2]] >= footprint_min)
        & (positions[:, [0, 2]] <= footprint_max),
        axis=1,
    )
    if not local.any():
        raise RuntimeError("cat footprint selected no top cushion particles")

    steps = int(round(args.duration / args.dt))
    output_steps = np.linspace(0, steps, args.output_frames, dtype=np.int64)
    lookup = {int(step): index for index, step in enumerate(output_steps)}
    snapshots = np.empty((args.output_frames, len(positions), 3), dtype=np.float32)
    offsets = np.empty(args.output_frames, dtype=np.float32)
    phases: list[str] = [""] * args.output_frames
    for step in range(steps + 1):
        time = step * args.dt
        offset_y, velocity_y, phase = collider_motion(
            time,
            args.press_travel,
            args.press_duration,
            args.hold_duration,
            args.release_duration,
        )
        if step in lookup:
            wp.synchronize_device(args.device)
            frame = lookup[step]
            snapshots[frame] = state.particle_x.numpy()
            offsets[frame] = offset_y
            phases[frame] = phase
        if step < steps:
            mesh_x = torch.as_tensor(
                cat_vertices0 + np.array([0.0, offset_y, 0.0], dtype=np.float32)
            )
            mesh_v = torch.zeros_like(mesh_x)
            mesh_v[:, 1] = velocity_y
            solver.p2g2p(model, state, args.dt, mesh_x=mesh_x, mesh_v=mesh_v, device=args.device)
    wp.synchronize_device(args.device)

    displacement = snapshots - snapshots[0]
    local_mean_down = -displacement[:, local, 1].mean(axis=1)
    local_max_down = -displacement[:, local, 1].min(axis=1)
    peak_frame = int(np.argmax(local_mean_down))
    peak_down = float(local_mean_down[peak_frame])
    final_down = float(local_mean_down[-1])
    rebound = peak_down - final_down
    summary = {
        "solver": "MPMAvatar warp_mpm (traditional cushion particles only)",
        "process": "rigid cat mesh press, hold, release, then free cushion rebound",
        "collider_mesh": "obj_0.glb",
        "cat_upside_down": True,
        "cushion_mesh": "obj_2.glb",
        "material": args.material,
        "yield_stress": args.yield_stress if args.material == "plastic" else None,
        "particle_count": len(positions),
        "cat_vertex_count": len(cat_vertices0),
        "cat_face_count": len(cat_faces),
        "selected_local_particle_count": int(local.sum()),
        "grid_resolution": args.grid_resolution,
        "grid_limit": args.grid_limit,
        "pitch": args.pitch,
        "dt": args.dt,
        "duration": args.duration,
        "initial_gap": args.initial_gap,
        "press_travel": args.press_travel,
        "press_duration": args.press_duration,
        "hold_duration": args.hold_duration,
        "release_duration": args.release_duration,
        "free_rebound_duration": args.duration - motion_end,
        "youngs_modulus": args.youngs_modulus,
        "friction": args.friction,
        "max_particle_displacement": float(np.linalg.norm(displacement, axis=2).max()),
        "peak_local_mean_downward_displacement": peak_down,
        "peak_local_max_downward_displacement": float(local_max_down.max()),
        "peak_compression_time": float(output_steps[peak_frame] * args.dt),
        "final_local_mean_downward_displacement": final_down,
        "recovered_local_mean_displacement": rebound,
        "rebound_fraction": float(rebound / peak_down) if peak_down > 0.0 else None,
        "final_mean_particle_speed": float(
            np.linalg.norm(state.particle_v.numpy(), axis=1).mean()
        ),
        "finite": bool(np.isfinite(snapshots).all()),
        "mesh_to_simulation_shift": shift.tolist(),
        "cat_alignment_after_scene_shift": alignment.tolist(),
        "volume_construction": "Y-up surface voxelization; fill each occupied XZ column to global bottom",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        particles=snapshots,
        cat_vertices_initial=cat_vertices0,
        cat_faces=cat_faces,
        cat_vertical_offsets=offsets,
        output_steps=output_steps,
        local_particle_mask=local,
        local_mean_downward_displacement=local_mean_down,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sampled_times = output_steps.astype(np.float64) * args.dt
    save_cross_section(
        args.output_dir / "cross_section.png", snapshots, cat_vertices0, cat_faces, offsets
    )
    save_response_curve(
        args.output_dir / "response_curve.png",
        sampled_times,
        local_mean_down,
        local_max_down,
        offsets,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
