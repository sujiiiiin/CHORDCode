#!/usr/bin/env python3
"""Apply a short local force pulse to the cushion, then let MPM evolve freely."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
        default=Path("trained/cat_with_cushion/mpmavatar_e1_impulse"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pitch", type=float, default=0.02)
    parser.add_argument("--grid-resolution", type=int, default=80)
    parser.add_argument("--grid-limit", type=float, default=1.6)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--duration", type=float, default=0.60)
    parser.add_argument("--output-frames", type=int, default=41)
    parser.add_argument("--force-duration", type=float, default=0.002)
    parser.add_argument("--force-magnitude", type=float, default=0.01)
    parser.add_argument("--force-radius", type=float, default=0.06)
    parser.add_argument("--diagnostic-steps", type=int, default=12)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    surface_mesh = load_chord_normalized_mesh(args.scene_dir)
    volume = heightfield_volume_particles(surface_mesh, args.pitch)
    dx = args.grid_limit / args.grid_resolution
    positions, shift = shift_into_mpm_domain(volume.positions, args.grid_limit, 4.0 * dx)
    state = initialize_traditional_state(
        positions, volume.volumes, args.grid_resolution, args.grid_limit, args.device
    )
    model = MPMModelStruct()
    model.init(len(positions), device=args.device, requires_grad=False)
    model.init_other_params(args.grid_resolution, args.grid_limit, device=args.device)
    solver = MPMWARP(
        len(positions), 0, 0, n_grid=args.grid_resolution,
        grid_lim=args.grid_limit, device=args.device,
    )
    solver.set_parameters_dict(
        model, state,
        {"material": "jelly", "g": [0.0, 0.0, 0.0], "density": args.density,
         "grid_v_damping_scale": 1.0, "rpic_damping": 0.0},
        device=args.device,
    )
    solver.set_E_nu(
        model, args.youngs_modulus, args.poisson_ratio, 0.0, 0.0,
        device=args.device,
    )
    solver.prepare_mu_lam(model, state, device=args.device)
    solver.add_surface_collider(
        [0.0, float(positions[:, 1].min() - 0.5 * args.pitch), 0.0],
        [0.0, 1.0, 0.0], surface="sticky", friction=0.0,
    )

    top_y = float(positions[:, 1].max())
    force_point = np.array(
        [np.median(positions[:, 0]), top_y, np.median(positions[:, 2])],
        dtype=np.float32,
    )
    region_half_size = np.array(
        [args.force_radius, args.pitch, args.force_radius], dtype=np.float32
    )
    region_center = force_point.copy()
    region_center[1] -= 0.5 * args.pitch
    # Match MPMAvatar's strict box selection exactly so total force is calibrated.
    selected = np.all(
        np.abs(positions - region_center[None]) < region_half_size[None], axis=1
    )
    if not selected.any():
        raise RuntimeError("force region selected no particles")
    # MPMAvatar's physical-force branch expects force per particle and divides by mass.
    force_vector = np.array([0.0, -args.force_magnitude, 0.0], dtype=np.float32)
    per_particle_force = force_vector / int(selected.sum())
    solver.add_impulse_on_particles(
        state, per_particle_force.tolist(), args.dt,
        point=region_center.tolist(), size=region_half_size.tolist(),
        num_dt=max(1, int(round(args.force_duration / args.dt))),
        device=args.device,
    )

    steps = int(round(args.duration / args.dt))
    output_steps = np.linspace(0, steps, args.output_frames, dtype=np.int64)
    lookup = {int(step): index for index, step in enumerate(output_steps)}
    snapshots = np.empty((args.output_frames, len(positions), 3), dtype=np.float32)
    for step in range(steps + 1):
        if step in lookup:
            wp.synchronize_device(args.device)
            snapshots[lookup[step]] = state.particle_x.numpy()
        if step < steps:
            diagnostic = step < args.diagnostic_steps
            if diagnostic:
                velocity = state.particle_v.numpy()
                print(
                    f"[runner diagnostic] step={step} before_p2g "
                    f"selected_v_mean={velocity[selected].mean(axis=0).tolist()} "
                    f"selected_v_max={np.linalg.norm(velocity[selected], axis=1).max():.9g}",
                    flush=True,
                )
            solver.p2g2p(
                model, state, args.dt, device=args.device, diagnostic=diagnostic
            )
    wp.synchronize_device(args.device)

    displacement = snapshots - snapshots[0]
    displacement_norm = np.linalg.norm(displacement, axis=2)
    sampled_times = output_steps.astype(np.float64) * args.dt
    sampled_max_displacement = displacement_norm.max(axis=1)
    sampled_selected_mean_down = -displacement[:, selected, 1].mean(axis=1)
    sampled_selected_max_down = -displacement[:, selected, 1].min(axis=1)
    peak_frame = int(np.argmax(sampled_max_displacement))
    domain_margin = 2.0 * dx + 1.0e-6
    clamped = np.any(
        (snapshots <= domain_margin)
        | (snapshots >= args.grid_limit - domain_margin),
        axis=2,
    )
    summary = {
        "solver": "MPMAvatar warp_mpm (traditional particles only)",
        "process": "local downward force pulse followed by free evolution",
        "material": "jelly (fixed-corotated elasticity)",
        "particle_count": len(positions),
        "grid_resolution": args.grid_resolution,
        "pitch": args.pitch,
        "dt": args.dt,
        "duration": args.duration,
        "force_point": force_point.tolist(),
        "force_direction": [0.0, -1.0, 0.0],
        "total_force": force_vector.tolist(),
        "total_impulse": (force_vector * args.force_duration).tolist(),
        "force_duration": args.force_duration,
        "force_radius": args.force_radius,
        "force_region_center": region_center.tolist(),
        "force_region_half_size": region_half_size.tolist(),
        "youngs_modulus": args.youngs_modulus,
        "selected_particle_count": int(selected.sum()),
        "max_particle_displacement": float(sampled_max_displacement[peak_frame]),
        "peak_displacement_time": float(sampled_times[peak_frame]),
        "peak_domain_clamped_particle_count": int(clamped.sum(axis=1).max()),
        "final_mean_particle_speed": float(
            np.linalg.norm(state.particle_v.numpy(), axis=1).mean()
        ),
        "finite": bool(np.isfinite(snapshots).all()),
        "mesh_to_simulation_shift": shift.tolist(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "trajectory.npz", particles=snapshots,
        force_point=force_point, force_direction=np.array([0.0, -1.0, 0.0]),
        force_mask=selected, mesh_shift=shift,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "frame_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frame", "time", "max_particle_displacement",
            "selected_mean_downward_displacement", "selected_max_downward_displacement",
        ])
        writer.writerows(zip(
            range(args.output_frames), sampled_times, sampled_max_displacement,
            sampled_selected_mean_down, sampled_selected_max_down,
        ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
