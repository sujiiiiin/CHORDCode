#!/usr/bin/env python3
"""E1 smoke test: MPMAvatar traditional cushion particles and a sphere mesh collider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments_mpm.e1.cushion_volume_particles import (
    heightfield_volume_particles,
    load_chord_normalized_mesh,
    shift_into_mpm_domain,
)
from warp_mpm.mpm_data_structure import (
    MPMModelStruct,
    MPMStateStruct,
    set_mat33_to_identity,
)
from warp_mpm.mpm_solver import MPMWARP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained/cat_with_cushion/mpmavatar_e1_smoke"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pitch", type=float, default=0.02)
    parser.add_argument("--grid-resolution", type=int, default=80)
    parser.add_argument("--grid-limit", type=float, default=1.6)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--duration", type=float, default=0.45)
    parser.add_argument("--output-frames", type=int, default=31)
    parser.add_argument("--sphere-radius", type=float, default=0.075)
    parser.add_argument("--sphere-travel", type=float, default=0.03)
    parser.add_argument("--press-duration", type=float, default=0.30)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    return parser.parse_args()


def collider_motion(time: float, travel: float, press_duration: float) -> tuple[float, float]:
    """Return downward offset and velocity for a press-then-hold trajectory."""
    if time < press_duration:
        return -travel * time / press_duration, -travel / press_duration
    return -travel, 0.0


def initialize_traditional_state(
    positions: np.ndarray,
    volumes: np.ndarray,
    grid_resolution: int,
    grid_limit: float,
    device: str,
) -> MPMStateStruct:
    """Initialize only MPMAvatar's traditional-particle branch."""
    count = len(positions)
    state = MPMStateStruct()
    state.init(count, n_elements=0, n_vertices=0, device=device, requires_grad=False)
    state.from_torch(
        torch.as_tensor(positions, dtype=torch.float32, device=device),
        torch.as_tensor(volumes, dtype=torch.float32, device=device),
        torch.empty((0, 3, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
        np.ones(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32),
        tensor_cov=torch.zeros((count, 6), dtype=torch.float32, device=device),
        n_grid=grid_resolution,
        grid_lim=grid_limit,
        device=device,
        requires_grad=False,
    )
    wp.launch(set_mat33_to_identity, count, [state.particle_F], device=device)
    wp.launch(set_mat33_to_identity, count, [state.particle_F_trial], device=device)
    return state


def save_cross_section(path: Path, snapshots: np.ndarray, sphere_centers: np.ndarray, radius: float) -> None:
    selected = np.linspace(0, len(snapshots) - 1, 4, dtype=int)
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), sharex=True, sharey=True)
    for axis, frame in zip(axes, selected):
        points = snapshots[frame]
        keep = np.abs(points[:, 2] - np.median(points[:, 2])) <= 0.015
        axis.scatter(points[keep, 0], points[keep, 1], s=3, color="#4c9f70")
        center = sphere_centers[frame]
        axis.add_patch(plt.Circle(center[:2], radius, fill=False, color="crimson", lw=2))
        axis.set_title(f"frame {frame}")
        axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    wp.init()
    mesh = load_chord_normalized_mesh(args.scene_dir)
    particles = heightfield_volume_particles(mesh, args.pitch)
    dx = args.grid_limit / args.grid_resolution
    positions, shift = shift_into_mpm_domain(particles.positions, args.grid_limit, 4.0 * dx)
    state = initialize_traditional_state(
        positions, particles.volumes, args.grid_resolution, args.grid_limit, args.device
    )
    model = MPMModelStruct()
    model.init(len(positions), device=args.device, requires_grad=False)
    model.init_other_params(args.grid_resolution, args.grid_limit, device=args.device)

    top_y = float(positions[:, 1].max())
    center0 = np.array(
        [np.median(positions[:, 0]), top_y + args.sphere_radius + 0.015, np.median(positions[:, 2])],
        dtype=np.float32,
    )
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=args.sphere_radius)
    sphere_vertices0 = np.asarray(sphere.vertices, dtype=np.float32) + center0
    sphere_faces = np.asarray(sphere.faces, dtype=np.int32)
    solver = MPMWARP(
        len(positions), 0, 0, n_grid=args.grid_resolution, grid_lim=args.grid_limit,
        mesh_vertices=sphere_vertices0, mesh_faces=sphere_faces, device=args.device,
    )
    solver.set_parameters_dict(
        model, state,
        {"material": "jelly", "g": [0.0, 0.0, 0.0], "density": args.density,
         "grid_v_damping_scale": 1.0, "rpic_damping": 0.0},
        device=args.device,
    )
    solver.set_E_nu(
        model, float(args.youngs_modulus), float(args.poisson_ratio), 0.0, 0.0,
        device=args.device,
    )
    solver.prepare_mu_lam(model, state, device=args.device)
    solver.add_surface_collider(
        [0.0, float(positions[:, 1].min() - 0.5 * args.pitch), 0.0],
        [0.0, 1.0, 0.0], surface="sticky", friction=0.0,
    )
    solver.add_mesh_collider(
        solver.mesh.id, n_grid=args.grid_resolution, friction=0.0, device=args.device
    )

    steps = int(round(args.duration / args.dt))
    output_steps = np.linspace(0, steps, args.output_frames, dtype=np.int64)
    lookup = {int(step): index for index, step in enumerate(output_steps)}
    snapshots = np.empty((args.output_frames, len(positions), 3), dtype=np.float32)
    centers = np.empty((args.output_frames, 3), dtype=np.float32)
    for step in range(steps + 1):
        time = step * args.dt
        offset_y, velocity_y = collider_motion(time, args.sphere_travel, args.press_duration)
        center = center0 + np.array([0.0, offset_y, 0.0], dtype=np.float32)
        if step in lookup:
            wp.synchronize_device(args.device)
            snapshots[lookup[step]] = state.particle_x.numpy()
            centers[lookup[step]] = center
        if step < steps:
            mesh_x = torch.as_tensor(
                sphere_vertices0 + np.array([0.0, offset_y, 0.0], dtype=np.float32),
                dtype=torch.float32,
            )
            mesh_v = torch.zeros_like(mesh_x)
            mesh_v[:, 1] = velocity_y
            solver.p2g2p(model, state, args.dt, mesh_x=mesh_x, mesh_v=mesh_v, device=args.device)
    wp.synchronize_device(args.device)

    displacement = snapshots - snapshots[0]
    penetration = np.maximum(
        args.sphere_radius
        - np.linalg.norm(snapshots - centers[:, None, :], axis=2),
        0.0,
    )
    final_top = snapshots[0, :, 1] >= np.quantile(snapshots[0, :, 1], 0.8)
    final_local = final_top & (
        np.linalg.norm(
            snapshots[0][:, [0, 2]] - centers[-1, [0, 2]][None], axis=1
        )
        <= args.sphere_radius
    )
    final_down = snapshots[0, final_local, 1] - snapshots[-1, final_local, 1]
    summary = {
        "solver": "MPMAvatar warp_mpm (traditional particles only)",
        "material": "jelly (fixed-corotated elasticity)",
        "particle_count": len(positions),
        "particle_volume": float(args.pitch**3),
        "n_elements": 0,
        "n_vertices": 0,
        "grid_resolution": args.grid_resolution,
        "grid_limit": args.grid_limit,
        "dt": args.dt,
        "sphere_radius": args.sphere_radius,
        "sphere_travel": args.sphere_travel,
        "max_particle_displacement": float(np.linalg.norm(displacement, axis=2).max()),
        "peak_particle_center_penetration": float(penetration.max()),
        "peak_inside_particle_count": int((penetration > 0.0).sum(axis=1).max()),
        "final_inside_particle_count": int((penetration[-1] > 0.0).sum()),
        "final_local_mean_downward_displacement": float(final_down.mean()),
        "final_local_max_downward_displacement": float(final_down.max()),
        "final_mean_particle_speed": float(np.linalg.norm(state.particle_v.numpy(), axis=1).mean()),
        "finite": bool(np.isfinite(snapshots).all()),
        "mesh_to_simulation_shift": shift.tolist(),
        "volume_construction": "Y-up surface voxelization; fill each occupied XZ column to global bottom",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "trajectory.npz", particles=snapshots, sphere_centers=centers)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_cross_section(args.output_dir / "cross_section.png", snapshots, centers, args.sphere_radius)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
