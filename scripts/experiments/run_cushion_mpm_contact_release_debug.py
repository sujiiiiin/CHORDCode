#!/usr/bin/env python3
"""E1-debug: controlled vertical contact followed by collider removal and release."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.run_cat_cushion_mpm_e1 import (
    build_volume_proxy,
    normalized_mesh,
    state_metrics,
)
from utils.mpm_utils import ColliderState, lame_parameters
from utils.warp_mpm import MPMConfig, WarpMPMSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=Path("data/cat_with_cushion"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proxy-pitch", type=float, default=0.015)
    parser.add_argument("--grid-dx", type=float, default=0.025)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--press-duration", type=float, default=0.25)
    parser.add_argument("--hold-duration", type=float, default=0.15)
    parser.add_argument("--press-depth", type=float, default=0.04)
    parser.add_argument("--initial-gap", type=float, default=0.01)
    parser.add_argument("--collider-radius", type=float, default=0.075)
    parser.add_argument("--output-frames", type=int, default=251)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--fixed-bottom-layers", type=int, default=2)
    parser.add_argument(
        "--particle-projection", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def contact_release_state(
    time: float,
    start_center: np.ndarray,
    press_depth: float,
    press_duration: float,
    hold_duration: float,
) -> tuple[ColliderState, bool]:
    """Prescribe a constant-speed press, hold, then remove the collider."""
    release_time = press_duration + hold_duration
    if press_duration <= 0.0 or hold_duration < 0.0:
        raise ValueError("press duration must be positive and hold duration non-negative")
    if time < press_duration:
        alpha = max(time, 0.0) / press_duration
        center = start_center + np.array([0.0, -press_depth * alpha, 0.0], dtype=np.float32)
        velocity = np.array([0.0, -press_depth / press_duration, 0.0], dtype=np.float32)
        return ColliderState(center.astype(np.float32), velocity), True
    center = start_center + np.array([0.0, -press_depth, 0.0], dtype=np.float32)
    return (
        ColliderState(center.astype(np.float32), np.zeros(3, dtype=np.float32)),
        time + 1.0e-12 < release_time,
    )


def main() -> None:
    args = parse_args()
    release_time = args.press_duration + args.hold_duration
    if args.duration <= release_time:
        raise ValueError("duration must exceed collider release time")
    rest = build_volume_proxy(normalized_mesh(args.scene_dir), args.proxy_pitch)
    bottom = float(rest[:, 1].min())
    fixed = (rest[:, 1] <= bottom + (args.fixed_bottom_layers - 0.5) * args.proxy_pitch).astype(np.int32)
    mu, lam = lame_parameters(args.youngs_modulus, args.poisson_ratio)
    config = MPMConfig(
        dx=args.grid_dx,
        dt=args.dt,
        density=args.density,
        particle_volume=args.proxy_pitch ** 3,
        mu=mu,
        lam=lam,
        damping=args.damping,
    )
    solver = WarpMPMSolver(rest, fixed, config, args.device)
    start_center = np.array(
        [np.median(rest[:, 0]), rest[:, 1].max() + args.collider_radius + args.initial_gap, 0.0],
        dtype=np.float32,
    )
    final_center = start_center + np.array([0.0, -args.press_depth, 0.0], dtype=np.float32)
    top = rest[:, 1] >= np.quantile(rest[:, 1], 0.8)
    local_top = top & (np.linalg.norm(rest[:, [0, 2]] - final_center[[0, 2]], axis=1) <= args.collider_radius)
    far = (np.linalg.norm(rest[:, [0, 2]] - final_center[[0, 2]], axis=1) >= 2.5 * args.collider_radius)
    far &= fixed == 0
    if not local_top.any() or not far.any():
        raise RuntimeError("failed to construct local and far masks")

    substeps = int(round(args.duration / args.dt))
    release_step = int(round(release_time / args.dt))
    output_steps = set(np.linspace(0, substeps, args.output_frames, dtype=np.int64).tolist())
    output_steps.add(release_step)
    rows = []
    for step in range(substeps + 1):
        time = step * args.dt
        collider, enabled = contact_release_state(
            time, start_center, args.press_depth, args.press_duration, args.hold_duration
        )
        if step in output_steps:
            wp.synchronize_device(args.device)
            rows.append({
                "step": step,
                "time": time,
                "collider_enabled": int(enabled),
                **state_metrics(
                    solver.positions(), solver.velocities(), solver.deformation_gradients(),
                    rest, fixed, local_top, far, collider.center, args.collider_radius, mu, lam,
                ),
            })
        if step < substeps:
            solver.step(
                collider.center, collider.velocity, args.collider_radius, 0.0, enabled,
                args.particle_projection,
            )

    release_row = min(rows, key=lambda row: abs(row["time"] - release_time))
    final_row = rows[-1]
    release_indent = release_row["local_mean_downward_displacement"]
    summary = {
        "particle_count": len(rest),
        "fixed_count": int(fixed.sum()),
        "dt": args.dt,
        "particle_projection": args.particle_projection,
        "release_time": release_time,
        "release_mean_indentation": release_indent,
        "release_rmse_from_rest": release_row["movable_rmse_from_rest"],
        "release_mean_frobenius_F_minus_I": release_row["mean_frobenius_F_minus_I"],
        "final_mean_indentation": final_row["local_mean_downward_displacement"],
        "final_to_release_indentation_ratio": (
            final_row["local_mean_downward_displacement"] / release_indent
            if release_indent > 0.0 else None
        ),
        "final_rmse_from_rest": final_row["movable_rmse_from_rest"],
        "final_mean_speed": final_row["mean_speed"],
        "final_mean_frobenius_F_minus_I": final_row["mean_frobenius_F_minus_I"],
        "final_mean_elastic_energy_density": final_row["mean_elastic_energy_density"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
