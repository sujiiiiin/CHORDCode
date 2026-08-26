"""Construct traditional MPM particles from the cat/cushion surface asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class VolumeParticles:
    positions: np.ndarray
    volumes: np.ndarray


def load_chord_normalized_mesh(scene_dir: Path, object_name: str = "obj_2.glb") -> trimesh.Trimesh:
    """Load one object with the same scene normalization used by CHORD."""
    scene = trimesh.load(scene_dir / "scene.glb", force="mesh", process=False)
    mesh = trimesh.load(scene_dir / object_name, force="mesh", process=False)
    center = scene.bounds.mean(axis=0)
    scale = 1.2 / np.ptp(scene.bounds, axis=0).max()
    mesh.vertices = (np.asarray(mesh.vertices) - center) * scale
    return mesh


def heightfield_volume_particles(mesh: trimesh.Trimesh, pitch: float) -> VolumeParticles:
    """Voxelize a Y-up cushion and fill every occupied XZ column to one base.

    ``obj_2.glb`` has an open underside, so generic watertight voxel filling is
    not valid.  The cushion-specific assumption is explicit: it is a solid
    height field with a flat bottom at the lowest observed voxel layer.
    """
    if pitch <= 0.0:
        raise ValueError("pitch must be positive")
    surface = np.asarray(mesh.voxelized(pitch).points, dtype=np.float64)
    if len(surface) == 0:
        raise ValueError("surface voxelization produced no points")
    origin = surface.min(axis=0)
    lattice = np.rint((surface - origin) / pitch).astype(np.int64)
    base_y = int(lattice[:, 1].min())
    column_tops: dict[tuple[int, int], int] = {}
    for ix, iy, iz in lattice:
        key = (int(ix), int(iz))
        column_tops[key] = max(column_tops.get(key, base_y), int(iy))
    filled = np.asarray(
        [(ix, iy, iz) for (ix, iz), top in column_tops.items() for iy in range(base_y, top + 1)],
        dtype=np.float64,
    )
    positions = (origin + pitch * filled).astype(np.float32)
    positions = np.unique(np.round(positions, decimals=7), axis=0).astype(np.float32)
    volumes = np.full(len(positions), pitch**3, dtype=np.float32)
    return VolumeParticles(positions=positions, volumes=volumes)


def shift_into_mpm_domain(
    positions: np.ndarray,
    grid_limit: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate without rescaling so every particle lies inside the MPM grid."""
    positions = np.asarray(positions, dtype=np.float32)
    shift = np.full(3, margin, dtype=np.float32) - positions.min(axis=0)
    shifted = positions + shift
    if np.any(shifted.max(axis=0) >= grid_limit - margin):
        raise ValueError("particle volume does not fit in the requested MPM domain")
    return shifted, shift
