import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


def _mesh_to_o3d_legacy(mesh):
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        o3_mesh = mesh
        if not o3_mesh.has_vertex_normals():
            o3_mesh.compute_vertex_normals()
        return o3_mesh

    if not hasattr(mesh, "v") or not hasattr(mesh, "f"):
        raise TypeError("Expected a mesh with attributes .v and .f")

    verts = mesh.v.detach().cpu().numpy().astype(np.float32, copy=False)
    faces = mesh.f.detach().cpu().numpy().astype(np.int32, copy=False)

    o3_mesh = o3d.geometry.TriangleMesh()
    o3_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3_mesh.triangles = o3d.utility.Vector3iVector(faces)
    try:
        o3_mesh.remove_degenerate_triangles()
        o3_mesh.remove_duplicated_vertices()
        o3_mesh.remove_duplicated_triangles()
    except Exception:
        pass
    o3_mesh.compute_vertex_normals()
    return o3_mesh


def sample_points_on_mesh(mesh, num_points: int, seed: int | None = None) -> torch.Tensor:
    if o3d is None:
        raise ImportError("Open3D is required for sampling points on mesh")
    if seed is not None:
        o3d.utility.random.seed(int(seed))

    o3_mesh = _mesh_to_o3d_legacy(mesh)
    pcd = o3_mesh.sample_points_uniformly(number_of_points=int(num_points))
    pts = np.asarray(pcd.points, dtype=np.float32)
    return torch.from_numpy(pts).cuda()
