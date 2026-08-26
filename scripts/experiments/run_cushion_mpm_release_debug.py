#!/usr/bin/env python3
"""E1-debug: release a known elastic compression without any collider."""

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

from scripts.experiments.run_cat_cushion_mpm_e1 import build_volume_proxy, normalized_mesh
from utils.mpm_utils import lame_parameters
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
    parser.add_argument("--output-frames", type=int, default=251)
    parser.add_argument("--compression", type=float, default=0.05)
    parser.add_argument("--youngs-modulus", type=float, default=1200.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.20)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--fixed-bottom-layers", type=int, default=2)
    return parser.parse_args()


def compressed_state(rest: np.ndarray, compression: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < compression < 1.0:
        raise ValueError("compression must lie in (0, 1)")
    scale_y = 1.0 - compression
    initial = rest.copy()
    bottom = float(rest[:, 1].min())
    initial[:, 1] = bottom + scale_y * (rest[:, 1] - bottom)
    deformation = np.tile(np.eye(3, dtype=np.float32), (len(rest), 1, 1))
    deformation[:, 1, 1] = scale_y
    return initial, deformation


def measure(
    positions: np.ndarray,
    velocities: np.ndarray,
    deformation: np.ndarray,
    rest: np.ndarray,
    movable: np.ndarray,
) -> dict[str, float]:
    displacement = positions[movable] - rest[movable]
    f_error = deformation[movable] - np.eye(3, dtype=np.float32)
    return {
        "movable_rmse_from_rest": float(np.sqrt(np.square(displacement).mean())),
        "mean_displacement_norm": float(np.linalg.norm(displacement, axis=1).mean()),
        "mean_speed": float(np.linalg.norm(velocities[movable], axis=1).mean()),
        "mean_frobenius_F_minus_I": float(np.linalg.norm(f_error, axis=(1, 2)).mean()),
        "mean_height_error": float(np.abs(displacement[:, 1]).mean()),
    }


def main() -> None:
    args = parse_args()
    if args.fixed_bottom_layers < 1:
        raise ValueError("fixed-bottom-layers must be at least one")
    rest = build_volume_proxy(normalized_mesh(args.scene_dir), args.proxy_pitch)
    bottom = float(rest[:, 1].min())
    fixed = (rest[:, 1] <= bottom + (args.fixed_bottom_layers - 0.5) * args.proxy_pitch).astype(np.int32)
    movable = fixed == 0
    initial, initial_f = compressed_state(rest, args.compression)
    # The prescribed bottom is the reference state, including at t=0.
    initial[~movable] = rest[~movable]
    initial_f[~movable] = np.eye(3, dtype=np.float32)
    mu, lam = lame_parameters(args.youngs_modulus, args.poisson_ratio)
    solver = WarpMPMSolver(
        rest,
        fixed,
        MPMConfig(
            dx=args.grid_dx,
            dt=args.dt,
            density=args.density,
            particle_volume=args.proxy_pitch ** 3,
            mu=mu,
            lam=lam,
            damping=args.damping,
        ),
        args.device,
        initial_positions=initial,
        initial_deformation=initial_f,
    )
    substeps = int(round(args.duration / args.dt))
    output_steps = set(np.linspace(0, substeps, args.output_frames, dtype=np.int64).tolist())
    rows = []
    disabled_center = np.zeros(3, dtype=np.float32)
    for step in range(substeps + 1):
        if step in output_steps:
            wp.synchronize_device(args.device)
            rows.append({
                "step": step,
                "time": step * args.dt,
                **measure(
                    solver.positions(), solver.velocities(), solver.deformation_gradients(), rest, movable
                ),
            })
        if step < substeps:
            solver.step(disabled_center, disabled_center, 0.0, 0.0, False, False)
    initial_error = rows[0]["movable_rmse_from_rest"]
    final_error = rows[-1]["movable_rmse_from_rest"]
    summary = {
        "particle_count": len(rest),
        "fixed_count": int(fixed.sum()),
        "dt": args.dt,
        "duration": args.duration,
        "compression": args.compression,
        "initial_rmse_from_rest": initial_error,
        "minimum_rmse_from_rest": min(row["movable_rmse_from_rest"] for row in rows),
        "final_rmse_from_rest": final_error,
        "final_to_initial_rmse_ratio": final_error / initial_error,
        "final_mean_speed": rows[-1]["mean_speed"],
        "final_mean_frobenius_F_minus_I": rows[-1]["mean_frobenius_F_minus_I"],
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
