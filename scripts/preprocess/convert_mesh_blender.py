#!/usr/bin/env python3
"""Convert a mesh into an initial 3D Gaussian point cloud.

The script mirrors the mesh-selection and normalization behavior used by the
Blender training entrypoints in this repository, but instead of optimizing
Gaussians from rendered images it initializes them directly from a densified
mesh surface. The output PLY follows this repo's 3DGS schema so it can be
loaded by ``scene.GaussianModel.load_ply``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814
SUPPORTED_SUFFIXES = {".glb", ".gltf", ".obj", ".ply"}
SKIP_STEMS = {"scene", "full_scene"}


@dataclass
class NormalizationRef:
    center: np.ndarray
    scale: float
    source_path: Path


@dataclass
class MeshJob:
    mesh_path: Path
    output_root: Path
    display_name: str


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def construct_attribute_names(sh_degree: int) -> List[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{i}" for i in range(3))
    rest_dim = 3 * ((sh_degree + 1) ** 2 - 1)
    names.extend(f"f_rest_{i}" for i in range(rest_dim))
    names.append("opacity")
    names.extend(f"scale_{i}" for i in range(3))
    names.extend(f"rot_{i}" for i in range(4))
    return names


def load_mesh_as_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, skip_materials=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError(f"Mesh '{path}' contains no geometry")
        dumped = list(loaded.dump())
        if not dumped:
            raise ValueError(f"Mesh '{path}' contains no renderable geometry")
        mesh = trimesh.util.concatenate(dumped)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh payload for '{path}': {type(loaded)!r}")

    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Mesh '{path}' has empty vertices or faces")
    return mesh


def extract_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = None
    vertex_colors = getattr(mesh.visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        colors = np.asarray(vertex_colors, dtype=np.float32)[..., :3]
    else:
        try:
            colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.float32)[..., :3]
        except Exception:
            colors = None

    if colors is None or len(colors) != len(mesh.vertices):
        return np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)

    if colors.size > 0 and float(colors.max()) > 1.0:
        colors = colors / 255.0
    return np.clip(colors.astype(np.float32), 0.0, 1.0)


def compute_center_and_scale(vertices: np.ndarray) -> Tuple[np.ndarray, float]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmax + vmin) * 0.5
    extent = float(np.max(vmax - vmin))
    scale = 1.2 / max(extent, 1e-8)
    return center.astype(np.float32), scale


def maybe_get_scene_normalization_ref(source_path: Path, self_norm: bool) -> Optional[NormalizationRef]:
    if self_norm:
        return None

    if source_path.is_dir():
        scene_path = source_path / "scene.glb"
    else:
        scene_path = source_path.parent / "scene.glb"
        if source_path.name == "scene.glb":
            scene_path = source_path

    if not scene_path.exists():
        return None

    scene_mesh = load_mesh_as_trimesh(scene_path)
    center, scale = compute_center_and_scale(np.asarray(scene_mesh.vertices))
    return NormalizationRef(center=center, scale=scale, source_path=scene_path)


def normalize_mesh(mesh: trimesh.Trimesh, colors: np.ndarray, ref: Optional[NormalizationRef]) -> Tuple[trimesh.Trimesh, np.ndarray]:
    mesh = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if ref is None:
        center, scale = compute_center_and_scale(vertices)
    else:
        center, scale = ref.center, ref.scale
    mesh.vertices = (vertices - center) * scale
    return mesh, colors.astype(np.float32)


def collect_mesh_jobs(args, source_root: Path) -> List[MeshJob]:
    if source_root.is_file():
        return [MeshJob(mesh_path=source_root, output_root=Path(args.model_path).resolve(), display_name=source_root.stem)]

    jobs: List[MeshJob] = []
    for root, dirs, files in os.walk(source_root):
        dirs.sort()
        files.sort()
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            stem = path.stem
            if stem in SKIP_STEMS:
                continue
            if not args.convert_all and stem != args.cur_train:
                continue
            jobs.append(
                MeshJob(
                    mesh_path=path,
                    output_root=Path(args.model_path).resolve() / stem,
                    display_name=stem,
                )
            )

    if jobs:
        return jobs

    available = sorted(
        p.stem
        for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and p.stem not in SKIP_STEMS
    )
    if args.convert_all:
        raise FileNotFoundError(f"No supported meshes found under '{source_root}'")
    raise FileNotFoundError(
        f"Could not find mesh '{args.cur_train}' under '{source_root}'. "
        f"Available mesh stems: {available[:30]}"
    )


def face_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    lengths = np.stack(
        [
            np.linalg.norm(v0 - v1, axis=1),
            np.linalg.norm(v1 - v2, axis=1),
            np.linalg.norm(v2 - v0, axis=1),
        ],
        axis=1,
    )
    return lengths


def estimate_auto_max_edge(mesh: trimesh.Trimesh, target_vertices: int) -> float:
    target_vertices = max(int(target_vertices), len(mesh.vertices), 1)
    surface_area = float(mesh.area)
    if not math.isfinite(surface_area) or surface_area <= 0.0:
        vmin = np.asarray(mesh.vertices).min(axis=0)
        vmax = np.asarray(mesh.vertices).max(axis=0)
        surface_area = float(np.prod(np.sort(vmax - vmin)[-2:]))
    edge = math.sqrt(max(surface_area, 1e-8) / target_vertices) * 1.6
    return max(edge, 1e-4)


def subdivide_long_edges_once(
    mesh: trimesh.Trimesh,
    colors: np.ndarray,
    max_edge_length: float,
    max_vertices: int,
) -> Tuple[trimesh.Trimesh, np.ndarray, bool]:
    vertices: List[List[float]] = np.asarray(mesh.vertices, dtype=np.float32).tolist()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors_list: List[List[float]] = np.asarray(colors, dtype=np.float32).tolist()
    # Keep texture coordinates through subdivision; interpolate UVs, not texels.
    source_uv = getattr(mesh.visual, "uv", None)
    uvs = np.asarray(source_uv, dtype=np.float64).tolist() if source_uv is not None else None

    changed = False
    new_faces: List[Tuple[int, int, int]] = []
    edge_cache = {}

    def midpoint_index(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        cached = edge_cache.get(key)
        if cached is not None:
            return cached

        vi = np.asarray(vertices[i], dtype=np.float32)
        vj = np.asarray(vertices[j], dtype=np.float32)
        ci = np.asarray(colors_list[i], dtype=np.float32)
        cj = np.asarray(colors_list[j], dtype=np.float32)

        vertices.append(((vi + vj) * 0.5).tolist())
        colors_list.append(((ci + cj) * 0.5).tolist())
        if uvs is not None:
            uvs.append(((np.asarray(uvs[i]) + np.asarray(uvs[j])) * 0.5).tolist())
        idx = len(vertices) - 1
        edge_cache[key] = idx
        return idx

    for face in faces:
        stack = [tuple(int(x) for x in face)]
        while stack:
            i0, i1, i2 = stack.pop()
            p0 = np.asarray(vertices[i0], dtype=np.float32)
            p1 = np.asarray(vertices[i1], dtype=np.float32)
            p2 = np.asarray(vertices[i2], dtype=np.float32)
            lengths = np.array(
                [
                    np.linalg.norm(p0 - p1),
                    np.linalg.norm(p1 - p2),
                    np.linalg.norm(p2 - p0),
                ],
                dtype=np.float32,
            )

            if float(lengths.max()) <= max_edge_length or len(vertices) >= max_vertices:
                new_faces.append((i0, i1, i2))
                continue

            changed = True
            edge_index = int(np.argmax(lengths))
            if edge_index == 0:
                mid = midpoint_index(i0, i1)
                stack.append((i0, mid, i2))
                stack.append((mid, i1, i2))
            elif edge_index == 1:
                mid = midpoint_index(i1, i2)
                stack.append((i1, mid, i0))
                stack.append((mid, i2, i0))
            else:
                mid = midpoint_index(i2, i0)
                stack.append((i2, mid, i1))
                stack.append((mid, i0, i1))

    mesh_new = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    if uvs is not None:
        mesh_new.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(uvs), material=mesh.visual.material,
        )
        colors_list = extract_vertex_colors(mesh_new)
    return mesh_new, np.asarray(colors_list, dtype=np.float32), changed


def densify_mesh(
    mesh: trimesh.Trimesh,
    colors: np.ndarray,
    target_vertices: int,
    max_edge_length: float,
    max_subdivide_iter: int,
    max_vertices: int,
) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    if max_edge_length > 0.0:
        working_edge = max_edge_length
    else:
        working_edge = estimate_auto_max_edge(mesh, target_vertices)

    for _ in range(max(1, max_subdivide_iter)):
        edge_lengths = face_edge_lengths(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        current_max = float(edge_lengths.max()) if edge_lengths.size > 0 else 0.0
        if len(mesh.vertices) >= target_vertices and current_max <= working_edge:
            break

        densified_mesh, densified_colors, changed = subdivide_long_edges_once(
            mesh,
            colors,
            max_edge_length=working_edge,
            max_vertices=max_vertices,
        )
        if changed:
            mesh, colors = densified_mesh, densified_colors
            continue

        if len(mesh.vertices) >= target_vertices:
            break
        working_edge *= 0.75
        if working_edge < 1e-6:
            break

    return mesh, colors.astype(np.float32), working_edge


def compute_vertex_edge_scales(vertices: np.ndarray, faces: np.ndarray, min_scale: float, scale_shrink: float) -> np.ndarray:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    ).astype(np.int64)
    if len(edges) == 0:
        return np.full((len(vertices), 3), min_scale, dtype=np.float32)

    deltas = np.abs(vertices[edges[:, 0]] - vertices[edges[:, 1]])
    accum = np.zeros((len(vertices), 3), dtype=np.float64)
    counts = np.zeros((len(vertices), 1), dtype=np.float64)

    np.add.at(accum, edges[:, 0], deltas)
    np.add.at(accum, edges[:, 1], deltas)
    np.add.at(counts[:, 0], edges[:, 0], 1.0)
    np.add.at(counts[:, 0], edges[:, 1], 1.0)

    valid = counts[:, 0] > 0.0
    accum[valid] /= counts[valid]
    accum[~valid] = min_scale

    scales = np.clip(accum / max(scale_shrink, 1e-6), min_scale, None)
    return scales.astype(np.float32)


def surface_scales_and_rotations(mesh, scales, tangent_factor, normal_ratio, robust=False):
    """Project the baseline covariance onto the tangent plane, then thin it."""
    from scipy.spatial.transform import Rotation
    normals = np.asarray(mesh.vertex_normals).copy()
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths < 1e-12] = [0, 0, 1]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    helper = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent = np.cross(helper, normals)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    basis = np.stack((tangent, np.cross(normals, tangent)), axis=-1)
    covariance = np.einsum('nki,nk,nkj->nij', basis, scales ** 2, basis)
    values, vectors = np.linalg.eigh(covariance)
    axes = basis @ vectors
    # Right-handed local frame, with its third axis along the mesh normal.
    first = axes[:, :, 0]
    rotation = np.stack((first, np.cross(normals, first), normals), axis=-1)
    tangent_scales = np.sqrt(np.maximum(values, 1e-20))
    result = np.column_stack((tangent_scales * tangent_factor,
                              tangent_scales.min(axis=1) * normal_ratio))
    if robust:
        # Unique topological neighbors, projected into the existing tangent frame.
        # Keep orientation and normal thickness identical to the surface baseline.
        edges = np.asarray(mesh.edges_unique)
        neighbors = [[] for _ in range(len(mesh.vertices))]
        for a, b in edges:
            delta = mesh.vertices[b] - mesh.vertices[a]
            for vertex, offset in ((a, delta), (b, -delta)):
                projected = offset @ rotation[vertex, :, :2]
                distance = np.linalg.norm(projected)
                if distance > 1e-12:
                    neighbors[vertex].append(distance)
        spacing = np.array([np.median(n) if n else 0.0 for n in neighbors])
        valid = spacing > 0
        if valid.any():
            # Object-local cap prevents sparse outliers from producing giant splats.
            spacing[valid] = np.minimum(spacing[valid], 2 * np.median(spacing[valid]))
            result[valid, :2] = spacing[valid, None] * tangent_factor
    quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    return result.astype(np.float32), quaternion.astype(np.float32)


def build_connected_vertices(vertices: np.ndarray, faces: np.ndarray) -> dict:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    unique_edges = np.unique(edges, axis=0)
    distances = np.linalg.norm(vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]], axis=1)

    adjacency = {int(i): {} for i in range(len(vertices))}
    for (i, j), dist in zip(unique_edges.tolist(), distances.tolist()):
        adjacency[int(i)][int(j)] = float(dist)
        adjacency[int(j)][int(i)] = float(dist)
    return adjacency


def write_gaussian_ply(
    ply_path: Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    sh_degree: int,
    init_opacity: float,
    rotations: Optional[np.ndarray] = None,
) -> None:
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    xyz = vertices.astype(np.float32)
    normals = np.zeros_like(xyz, dtype=np.float32)
    f_dc = rgb_to_sh(colors.astype(np.float32)).astype(np.float32)
    rest_dim = 3 * ((sh_degree + 1) ** 2 - 1)
    f_rest = np.zeros((len(vertices), rest_dim), dtype=np.float32)
    opacity = inverse_sigmoid(np.full((len(vertices), 1), init_opacity, dtype=np.float32)).astype(np.float32)
    scale = np.log(scales.astype(np.float32))
    rotation = np.zeros((len(vertices), 4), dtype=np.float32)
    rotation[:, 0] = 1.0
    if rotations is not None:
        rotation = rotations

    dtype_full = [(name, "f4") for name in construct_attribute_names(sh_degree)]
    elements = np.empty(len(vertices), dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(ply_path)


def save_densified_mesh(mesh: trimesh.Trimesh, colors: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh_to_save = mesh.copy()
    rgba = np.concatenate(
        [
            np.clip(np.round(colors * 255.0), 0.0, 255.0).astype(np.uint8),
            np.full((len(colors), 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )
    mesh_to_save.visual = trimesh.visual.ColorVisuals(mesh=mesh_to_save, vertex_colors=rgba)
    mesh_to_save.export(path)


def convert_single_mesh(args, job: MeshJob, norm_ref: Optional[NormalizationRef]) -> None:
    mesh = load_mesh_as_trimesh(job.mesh_path)
    colors = extract_vertex_colors(mesh)
    mesh, colors = normalize_mesh(mesh, colors, norm_ref)

    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    mesh, colors, final_edge_target = densify_mesh(
        mesh=mesh,
        colors=colors,
        target_vertices=args.target_vertices,
        max_edge_length=args.max_edge_length,
        max_subdivide_iter=args.max_subdivide_iter,
        max_vertices=args.max_vertices,
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scales = compute_vertex_edge_scales(
        vertices=vertices,
        faces=faces,
        min_scale=args.min_scale,
        scale_shrink=args.scale_shrink,
    )

    rotations = None
    if args.shape_mode in ("surface", "surface_robust"):
        scales, rotations = surface_scales_and_rotations(
            mesh, scales, args.tangent_factor, args.normal_ratio,
            robust=args.shape_mode == "surface_robust")
        scales = np.maximum(scales, args.min_scale)

    point_cloud_dir = job.output_root / "point_cloud" / f"iteration_{args.output_iteration}"
    ply_path = point_cloud_dir / "point_cloud.ply"
    write_gaussian_ply(
        ply_path=ply_path,
        vertices=vertices,
        colors=colors,
        scales=scales,
        sh_degree=args.sh_degree,
        init_opacity=args.init_opacity,
        rotations=rotations,
    )

    if not args.skip_connected_vertices:
        adjacency = build_connected_vertices(vertices, faces)
        connected_path = job.output_root / "connected_vertices.json"
        connected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(connected_path, "w", encoding="utf-8") as f:
            json.dump(adjacency, f, indent=2)

    if args.save_densified_mesh:
        save_densified_mesh(mesh, colors, job.output_root / "densified_mesh.ply")

    print(
        f"[Convert] {job.display_name}: verts {original_vertices} -> {len(vertices)}, "
        f"faces {original_faces} -> {len(faces)}, target_edge={final_edge_target:.6f}"
    )
    print(f"[Convert] Wrote Gaussian PLY to {ply_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a mesh or Blender mesh directory into an initial 3D Gaussian point cloud."
    )
    parser.add_argument("--mesh_source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cur_train", type=str, default="obj_0")
    parser.add_argument("--self_norm", action="store_true", default=False)
    parser.add_argument("--convert_all", action="store_true", default=False)
    parser.add_argument("--output_iteration", type=int, default=0)
    parser.add_argument("--sh_degree", type=int, default=3)

    parser.add_argument("--shape_mode", choices=["legacy", "surface", "surface_robust"], default="legacy")
    parser.add_argument("--tangent_factor", type=float, default=1.0)
    parser.add_argument("--normal_ratio", type=float, default=0.1)
    parser.add_argument("--target_vertices", type=int, default=30_000)
    parser.add_argument("--max_vertices", type=int, default=100_000)
    parser.add_argument("--max_edge_length", type=float, default=0.0)
    parser.add_argument("--max_subdivide_iter", type=int, default=8)
    parser.add_argument("--scale_shrink", type=float, default=0.9)
    parser.add_argument("--min_scale", type=float, default=1e-4)
    parser.add_argument("--init_opacity", type=float, default=0.99999)
    parser.add_argument("--save_densified_mesh", action="store_true", default=False)
    parser.add_argument("--skip_connected_vertices", action="store_true", default=False)

    parser.add_argument("--bake_meshes", action="store_true", default=False)
    parser.add_argument("--bake_only_cur_train", type=str, default="")
    parser.add_argument("--blender_env_exr", type=str, default="")
    parser.add_argument(
        "--bake_script_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "bake_mesh_blender.py"),
    )
    parser.add_argument("--baked_mesh_root", type=str, default="")
    parser.add_argument("--force_rebake", action="store_true", default=False)
    parser.add_argument("--bake_texture_size", type=int, default=0)
    parser.add_argument("--bake_samples", type=int, default=256)
    parser.add_argument("--bake_margin", type=int, default=16)
    parser.add_argument("--bake_world_strength", type=float, default=1.0)
    parser.add_argument("--bake_device", type=str, default="AUTO", choices=["AUTO", "CUDA", "CPU"])
    parser.add_argument("--bake_alpha_threshold", type=float, default=0.999)

    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> None:
    args = parse_args(argv)

    source_root = Path(args.mesh_source_path).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"mesh_source_path does not exist: {source_root}")

    if args.bake_meshes:
        from train_gaussian_for_scene_blender_fast import prepare_baked_mesh_source

        if not source_root.is_dir():
            raise ValueError("--bake_meshes currently requires --mesh_source_path to be a directory")
        if not args.blender_env_exr:
            raise ValueError("--blender_env_exr is required when --bake_meshes is enabled")
        if not args.convert_all and not str(args.bake_only_cur_train).strip():
            args.bake_only_cur_train = args.cur_train
        source_root = Path(prepare_baked_mesh_source(args)).resolve()

    jobs = collect_mesh_jobs(args, source_root)
    norm_ref = maybe_get_scene_normalization_ref(source_root, args.self_norm)
    if norm_ref is not None:
        print(f"[Normalize] Using scene reference from {norm_ref.source_path}")
    else:
        print("[Normalize] Using per-mesh normalization")

    for job in jobs:
        convert_single_mesh(args, job, norm_ref)


if __name__ == "__main__":
    main(sys.argv[1:])
