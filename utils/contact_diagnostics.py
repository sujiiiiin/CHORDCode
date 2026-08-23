"""Geometry-only diagnostics for contact events in deformed object trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceSamples:
    face_indices: np.ndarray
    barycentric: np.ndarray


def select_low_vertices(vertices: np.ndarray, quantile: float) -> np.ndarray:
    """Select a deterministic low-height candidate set (the initial foot proxy)."""
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    threshold = np.quantile(vertices[:, 1], quantile)
    return np.flatnonzero(vertices[:, 1] <= threshold)


def label_vertex_components(
    faces: np.ndarray,
    vertex_indices: np.ndarray,
    vertex_count: int,
) -> tuple[int, np.ndarray]:
    """Label mesh-connected components within a selected vertex subset."""
    faces = np.asarray(faces, dtype=np.int64)
    vertex_indices = np.asarray(vertex_indices, dtype=np.int64)
    selected = np.zeros(vertex_count, dtype=bool)
    selected[vertex_indices] = True
    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]), axis=0)
    edges = edges[selected[edges].all(axis=1)]
    adjacency = sparse.coo_matrix(
        (
            np.ones(len(edges) * 2, dtype=np.uint8),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    return connected_components(adjacency[vertex_indices][:, vertex_indices])


def sample_surface_template(
    vertices: np.ndarray,
    faces: np.ndarray,
    sample_count: int,
    seed: int,
) -> SurfaceSamples:
    """Sample reusable face ids and barycentric weights from a rest mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 0.0)
    if not valid.any():
        raise ValueError("mesh has no non-degenerate faces")
    probabilities = np.where(valid, areas, 0.0)
    probabilities /= probabilities.sum()

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=sample_count, p=probabilities)
    uv = rng.random((sample_count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    barycentric = np.column_stack((1.0 - uv.sum(axis=1), uv))
    return SurfaceSamples(face_indices=face_indices, barycentric=barycentric)


def deform_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    samples: SurfaceSamples,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate sampled surface points and their current face normals."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces[samples.face_indices]]
    points = (triangles * samples.barycentric[:, :, None]).sum(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    return points, normals


def frame_contact_metrics(
    actor_vertices: np.ndarray,
    candidate_indices: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    contact_threshold: float,
) -> dict[str, float | int]:
    """Measure unsigned proximity plus an oriented-gap proxy for one frame."""
    actor_vertices = np.asarray(actor_vertices, dtype=np.float64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidates = actor_vertices[candidate_indices]
    tree = cKDTree(surface_points)
    candidate_distances, nearest = tree.query(candidates, k=1)
    all_distances, _ = tree.query(actor_vertices, k=1)

    offsets = candidates - surface_points[nearest]
    oriented_gaps = np.einsum("ij,ij->i", offsets, surface_normals[nearest])
    near = candidate_distances <= contact_threshold
    near_gaps = oriented_gaps[near]
    negative_near_gaps = near_gaps[near_gaps < 0.0]

    return {
        "all_min_distance": float(all_distances.min()),
        "candidate_min_distance": float(candidate_distances.min()),
        "candidate_mean_distance": float(candidate_distances.mean()),
        "near_candidate_count": int(near.sum()),
        "near_candidate_fraction": float(near.mean()),
        # This is only a proxy on non-watertight meshes and depends on face orientation.
        "min_oriented_gap_proxy": float(oriented_gaps.min()),
        "mean_near_oriented_gap_proxy": (
            float(near_gaps.mean()) if len(near_gaps) else float("nan")
        ),
        "negative_near_gap_count": int(len(negative_near_gaps)),
        "negative_near_gap_mean_depth_proxy": (
            float((-negative_near_gaps).mean()) if len(negative_near_gaps) else 0.0
        ),
    }


def infer_contact_windows(
    near_counts: np.ndarray,
    padding: int = 2,
    min_count: int = 1,
) -> list[tuple[int, int]]:
    """Return inclusive, padded windows for consecutive near-contact frames."""
    near_counts = np.asarray(near_counts)
    if near_counts.ndim != 1:
        raise ValueError("near_counts must be one-dimensional")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    active = np.flatnonzero(near_counts >= min_count)
    if not len(active):
        return []

    runs: list[tuple[int, int]] = []
    start = previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))

    last_frame = len(near_counts) - 1
    padded = [(max(0, start - padding), min(last_frame, end + padding)) for start, end in runs]
    merged: list[tuple[int, int]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
