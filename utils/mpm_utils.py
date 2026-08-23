"""Small, backend-independent helpers for the cat/cushion MPM experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColliderState:
    center: np.ndarray
    velocity: np.ndarray


def column_fill_points(
    surface_points: np.ndarray,
    pitch: float,
    fill_to_global_bottom: bool = False,
) -> np.ndarray:
    """Build a cushion-specific volume proxy by filling vertical surface columns.

    This intentionally uses the known cushion-up axis (Y). It is robust to the
    holes in ``obj_2.glb`` but is not a general mesh repair algorithm.
    """
    points = np.asarray(surface_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface_points must have shape [N, 3]")
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    origin = points.min(axis=0)
    indices = np.rint((points - origin) / pitch).astype(np.int64)
    columns: dict[tuple[int, int], list[int]] = {}
    for ix, iy, iz in indices:
        columns.setdefault((int(ix), int(iz)), []).append(int(iy))
    global_bottom = int(indices[:, 1].min())
    filled = []
    for (ix, iz), ys in columns.items():
        bottom = global_bottom if fill_to_global_bottom else min(ys)
        for iy in range(bottom, max(ys) + 1):
            filled.append((ix, iy, iz))
    filled_indices = np.asarray(filled, dtype=np.float64)
    return (origin + filled_indices * pitch).astype(np.float32)


def collider_state(
    time: float,
    start_center: np.ndarray,
    press_depth: float,
    slide_distance: float,
    duration: float,
) -> ColliderState:
    """Piecewise-linear hover, press, slide, lift, and recovery trajectory."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    start = np.asarray(start_center, dtype=np.float64)
    u = np.clip(time / duration, 0.0, 1.0)
    # phase endpoints: hover, press, slide, lift, recovery
    knots = np.array([0.0, 0.15, 0.40, 0.60, 0.80, 1.0])
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -press_depth, 0.0],
            [slide_distance, -press_depth, 0.0],
            [slide_distance, 0.0, 0.0],
            [slide_distance, 0.0, 0.0],
        ]
    )
    phase = min(np.searchsorted(knots, u, side="right") - 1, len(knots) - 2)
    phase = max(phase, 0)
    width = knots[phase + 1] - knots[phase]
    alpha = (u - knots[phase]) / width
    offset = (1.0 - alpha) * offsets[phase] + alpha * offsets[phase + 1]
    velocity = (offsets[phase + 1] - offsets[phase]) / (width * duration)
    if u >= 1.0:
        velocity[:] = 0.0
    return ColliderState((start + offset).astype(np.float32), velocity.astype(np.float32))


def lame_parameters(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    if youngs_modulus <= 0.0:
        raise ValueError("youngs_modulus must be positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must be in (-1, 0.5)")
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    return mu, lam
