import os

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

# --- helpers ---------------------------------------------------------------
def _mesh_to_o3d_legacy(mesh):
    """Convert a user mesh object with attributes .v (Vx3) and .f (Fx3 or Fxn)
    into an Open3D legacy TriangleMesh. Attempts to be robust to common issues:
      - 1-based face indices (OBJ-style)
      - Non-tri faces (simple fan triangulation)
      - Normalization: if the mesh object has `ori_center` / `ori_scale` and
        a boolean `restore_world` attribute, undo normalization.
      - Avoid *destructive* topology edits (do NOT remove non-manifold edges).
    """
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        o3 = mesh
        if not o3.has_vertex_normals():
            o3.compute_vertex_normals()
        return o3

    # 1) Fetch vertices / faces
    if not (hasattr(mesh, 'v') and hasattr(mesh, 'f')):
        raise TypeError("Unsupported mesh type for _mesh_to_o3d_legacy: expected Open3D TriangleMesh or object with .v and .f")

    V = mesh.v.detach().cpu().numpy() if hasattr(mesh.v, 'detach') else np.asarray(mesh.v)
    F = mesh.f.detach().cpu().numpy() if hasattr(mesh.f, 'detach') else np.asarray(mesh.f)

    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Vertices must be (N,3); got {V.shape}")
    if F.ndim != 2 or F.shape[1] < 3:
        raise ValueError(f"Faces must be (M,>=3); got {F.shape}")

    V = V.astype(np.float32, copy=True)
    F = F.astype(np.int64, copy=True)

    # 2) Undo normalization if requested by the mesh object
    if hasattr(mesh, 'ori_center') and hasattr(mesh, 'ori_scale') and getattr(mesh, 'restore_world', False):
        try:
            V = V * float(mesh.ori_scale) + np.asarray(mesh.ori_center, dtype=np.float32)
        except Exception:
            pass

    # 3) Fix 1-based indexing if detected
    fmin = F.min()
    fmax = F.max()
    if fmin == 1 and fmax <= V.shape[0]:
        F = F - 1
    elif fmin < 0 or fmax >= V.shape[0]:
        # Replace out-of-range indices by clipping; better than crashing, but warn.
        F = np.clip(F, 0, V.shape[0]-1)

    # 4) Triangulate if needed (fan triangulation)
    if F.shape[1] != 3:
        tris = []
        for face in F:
            v0 = face[0]
            for k in range(1, face.shape[0]-1):
                tris.append([v0, face[k], face[k+1]])
        F = np.asarray(tris, dtype=np.int64)

    # 5) Build Open3D mesh
    o3 = o3d.geometry.TriangleMesh()
    o3.vertices = o3d.utility.Vector3dVector(V)
    o3.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))

    # 6) Optional normals
    if hasattr(mesh, 'vn') and mesh.vn is not None:
        VN = mesh.vn.detach().cpu().numpy() if hasattr(mesh.vn, 'detach') else np.asarray(mesh.vn)
        if VN.shape[0] == V.shape[0] and VN.shape[1] == 3:
            o3.vertex_normals = o3d.utility.Vector3dVector(VN.astype(np.float32))
        else:
            o3.compute_vertex_normals()
    else:
        o3.compute_vertex_normals()

    # 7) Minimal cleanup only — avoid opening holes
    # Keep degenerate/duplicate removal, but DO NOT remove non-manifold edges by default.
    try:
        o3.remove_degenerate_triangles()
        o3.remove_duplicated_vertices()
        o3.remove_duplicated_triangles()
        # o3.remove_non_manifold_edges()  # can open seams; skip by default
    except Exception:
        pass

    # 8) Diagnostics (optional): print basic stats & watertightness if env asks
    if os.environ.get('VOXEL_VERBOSE', '0') == '1':
        try:
            print("[voxel] verts:", np.asarray(o3.vertices).shape[0],
                  "tris:", np.asarray(o3.triangles).shape[0],
                  "watertight:", o3.is_watertight(),
                  "self_intersect:", o3.is_self_intersecting())
        except Exception:
            pass

    return o3

# --- topology helpers -----------------------------------------------------

def _cuda_device() -> torch.device:
    return torch.device("cuda")


def _build_voxel_xyz(
    x_grid_min: int,
    x_grid_max: int,
    y_grid_min: int,
    y_grid_max: int,
    z_grid_min: int,
    z_grid_max: int,
    voxel_size: float,
) -> torch.Tensor:
    device = _cuda_device()
    Nx = x_grid_max - x_grid_min
    Ny = y_grid_max - y_grid_min
    Nz = z_grid_max - z_grid_min
    voxel_xyz = torch.empty((Nx, Ny, Nz, 3), dtype=torch.float32, device=device)
    voxel_xyz[:, :, :, 0] = (
        torch.arange(x_grid_min, x_grid_max, dtype=torch.float32, device=device).view(-1, 1, 1) * voxel_size
        + 0.5 * voxel_size
    )
    voxel_xyz[:, :, :, 1] = (
        torch.arange(y_grid_min, y_grid_max, dtype=torch.float32, device=device).view(1, -1, 1) * voxel_size
        + 0.5 * voxel_size
    )
    voxel_xyz[:, :, :, 2] = (
        torch.arange(z_grid_min, z_grid_max, dtype=torch.float32, device=device).view(1, 1, -1) * voxel_size
        + 0.5 * voxel_size
    )
    return voxel_xyz.reshape(-1, 3)


def _conv3d_mask(input_tensor: torch.Tensor, kernel: torch.Tensor, padding: int = 1) -> torch.Tensor:
    if input_tensor.is_cuda:
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv3d(input_tensor, kernel, padding=padding)
    return F.conv3d(input_tensor, kernel, padding=padding)


def _flood_fill_outside(free_mask: torch.Tensor, max_iter: int | None = None) -> torch.Tensor:
    """Mark OUTSIDE voxels by flood-filling from the grid boundary through free cells.
    Args:
        free_mask: (Nx,Ny,Nz) bool tensor where True means passable (air).
        max_iter: optional cap on iterations; default Nx+Ny+Nz.
    Returns:
        outside: (Nx,Ny,Nz) bool tensor of voxels reachable from outside.
    """
    if free_mask.dtype != torch.bool:
        free_mask = free_mask > 0
    assert free_mask.dim() == 3, "free_mask must be (Nx,Ny,Nz)"
    Nx, Ny, Nz = free_mask.shape

    # Seed outside at *free* boundary cells
    outside = torch.zeros_like(free_mask)
    outside[0, :, :]  = free_mask[0, :, :]
    outside[-1, :, :] = free_mask[-1, :, :]
    outside[:, 0, :]  = free_mask[:, 0, :]
    outside[:, -1, :] = free_mask[:, -1, :]
    outside[:, :, 0]  = free_mask[:, :, 0]
    outside[:, :, -1] = free_mask[:, :, -1]

    # 6-neighborhood kernel (no diagonals) to avoid directional bias
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32, device=free_mask.device)
    kernel[0, 0, 1, 1, 0] = 1
    kernel[0, 0, 1, 1, 2] = 1
    kernel[0, 0, 1, 0, 1] = 1
    kernel[0, 0, 1, 2, 1] = 1
    kernel[0, 0, 0, 1, 1] = 1
    kernel[0, 0, 2, 1, 1] = 1

    out_b  = outside.unsqueeze(0).unsqueeze(0)  # (1,1,Nx,Ny,Nz) bool
    free_b = free_mask.unsqueeze(0).unsqueeze(0)

    if max_iter is None:
        max_iter = int(Nx + Ny + Nz)  # generous upper bound

    for _ in range(max_iter):
        prev = out_b
        # one dilation step
        dil = (_conv3d_mask(out_b.float(), kernel, padding=1) > 0)
        # only expand into free cells
        cand = dil & free_b
        out_b = prev | cand
        if torch.equal(out_b, prev):
            break

    return out_b[0, 0]

class VoxelGrid:
    def __init__(self, base_voxel_size, occupy_K=10, occupy_threshold = 0.1):
        self.base_voxel_size = base_voxel_size
        self.occupy_K = occupy_K
        self.occupy_threshold = occupy_threshold
    
    def init_from_mesh(self, mesh, padding=0.0, relative_padding=False, chunk_size=2000000, return_sdf=False
                       , sdf_band=0.0, close_iters=0, fill_mode="unsigned", surface_only=True, surface_band=None, br_range=0.75):
        """
        Initialize the voxel grid (centers) from a triangle mesh and compute an
        occupancy mask using Open3D's RaycastingScene signed or unsigned distances.

        Args:
            mesh: Either an Open3D TriangleMesh (legacy) or an object with
                  attributes .v (Vx3) and .f (Fx3) as in your previous loader.
            padding (float): Extra padding around the mesh AABB. If
                  relative_padding=True, interpreted as a fraction of the AABB size.
            relative_padding (bool): Interpret `padding` as relative fraction.
            chunk_size (int): Number of query points per batch for SDF evaluation.
            return_sdf (bool): If True, also return the SDF grid as a torch tensor
                  shaped (Nx, Ny, Nz).
            sdf_band (float): Thickness (in world units) for the surface band. Used
                  as a barrier thickness in unsigned mode and tolerance in signed mode.
            close_iters (int): Morphological closing iterations to seal tiny cracks.
            fill_mode (str): One of {"unsigned", "signed"}. "signed" (default)
                  classifies inside voxels directly using SDF < 0 with optional
                  morphological closing. "unsigned" builds a barrier from unsigned
                  distance <= sdf_band and flood-fills outside (robust for open meshes).
            surface_only (bool): If True, only keep voxels within a thin band near
                  the surface. The band thickness is `surface_band` if provided,
                  otherwise it defaults to `max(sdf_band, 0.75*voxel_size)` in unsigned
                  mode and to `max(sdf_band, 0.25*voxel_size)` in signed mode.
            surface_band (float or None): Explicit surface band thickness in world
                  units when `surface_only=True`.

        Returns:
            occupy_xyz (torch.FloatTensor): (M,3) centers of occupied voxels.
            sdf_grid (optional torch.FloatTensor): (Nx, Ny, Nz) SDF values.
        """
        # 1) Convert to Open3D legacy mesh
        o3_mesh = _mesh_to_o3d_legacy(mesh)

        # 2) Bounds with optional padding
        aabb = o3_mesh.get_axis_aligned_bounding_box()
        # Make explicit writable copies; Open3D can return read-only buffers
        bmin = np.array(aabb.min_bound, dtype=np.float64, copy=True)
        bmax = np.array(aabb.max_bound, dtype=np.float64, copy=True)
        if relative_padding:
            pad = (bmax - bmin) * float(padding)
            bmin -= pad
            bmax += pad
        else:
            p = float(padding)
            bmin -= p
            bmax += p

        # 3) Convert world bounds -> integer grid bounds (aligned to base_voxel_size)
        vx = float(self.base_voxel_size)
        x_grid_min = int(np.floor(bmin[0] / vx))
        y_grid_min = int(np.floor(bmin[1] / vx))
        z_grid_min = int(np.floor(bmin[2] / vx))
        x_grid_max = int(np.ceil (bmax[0] / vx))
        y_grid_max = int(np.ceil (bmax[1] / vx))
        z_grid_max = int(np.ceil (bmax[2] / vx))
        self.grid_min = (x_grid_min, y_grid_min, z_grid_min)
        self.grid_max = (x_grid_max, y_grid_max, z_grid_max)

        Nx = x_grid_max - x_grid_min
        Ny = y_grid_max - y_grid_min
        Nz = z_grid_max - z_grid_min
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError("Invalid grid dimensions computed from mesh bounds.")

        self.voxel_xyz = _build_voxel_xyz(
            x_grid_min,
            x_grid_max,
            y_grid_min,
            y_grid_max,
            z_grid_min,
            z_grid_max,
            vx,
        )


        # 5) Build Open3D RaycastingScene
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        # 6) Compute (signed and/or unsigned) distance at voxel centers in chunks
        pts_np = self.voxel_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        N = pts_np.shape[0]
        # Always prepare arrays; some modes use both (hybrid logic for robustness)
        dist_all = None
        sdf_all = None
        want_unsigned = (fill_mode == "unsigned") or (fill_mode == "signed")
        want_signed = (fill_mode == "signed")
        if want_unsigned:
            dist_all = np.empty((N,), dtype=np.float32)
        if want_signed:
            sdf_all = np.empty((N,), dtype=np.float32)

        chunk_size = int(chunk_size)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            pts_chunk = o3d.core.Tensor(pts_np[start:end], dtype=o3d.core.Dtype.Float32)
            if want_unsigned:
                d = scene.compute_distance(pts_chunk).numpy()
                dist_all[start:end] = d
            if want_signed:
                s = scene.compute_signed_distance(pts_chunk).numpy()
                sdf_all[start:end] = s

        # 7) Occupancy mask from unsigned or signed distance, with optional closing
        if fill_mode == "unsigned":
            # Barrier from unsigned distance <= tau, then outside flood-fill
            tau = max(float(sdf_band), 0.75 * vx)  # thicker barrier for robustness
            dist_grid = torch.from_numpy(dist_all.reshape(Nx, Ny, Nz)).to(device=self.voxel_xyz.device, dtype=torch.float32)
            # Optional closing on the barrier to seal tiny cracks
            barrier = (dist_grid <= tau).float().unsqueeze(0).unsqueeze(0)
            if int(close_iters) > 0:
                kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=barrier.device)
                for _ in range(int(close_iters)):
                    barrier = (_conv3d_mask(barrier, kernel, padding=1) > 0).float()
                    barrier = (_conv3d_mask(barrier, kernel, padding=1) == 27).float()
            barrier = (barrier[0,0] > 0)
            free = ~barrier
            outside = _flood_fill_outside(free)
            occ_mask = (~outside)
            if surface_only:
                band = float(surface_band) if surface_band is not None else max(float(sdf_band), br_range * vx)
                near = dist_grid <= band
                occ_mask = occ_mask & near
        else:
            # Signed mode (hybrid):
            #   1) Build robust interior via unsigned barrier + flood fill (no reliance on sign).
            #   2) Optionally constrain with SDF<0 to remove internal voids when signs are reliable.
            #   3) Apply near-surface band if requested.
            assert dist_all is not None and sdf_all is not None
            dist_grid = torch.from_numpy(dist_all.reshape(Nx, Ny, Nz)).to(device=self.voxel_xyz.device, dtype=torch.float32)
            sdf_grid = torch.from_numpy(sdf_all.reshape(Nx, Ny, Nz)).to(device=self.voxel_xyz.device, dtype=torch.float32)

            # Step 1: interior via barrier on unsigned distance
            # Use a slightly thicker barrier when no explicit closing is requested
            # to reduce leakage through discretization gaps on coarse grids.
            tau_default = br_range * vx if int(close_iters) == 0 else 0.75 * vx
            tau = max(float(sdf_band), tau_default)
            barrier = (dist_grid <= tau).float().unsqueeze(0).unsqueeze(0)
            if int(close_iters) > 0:
                kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=barrier.device)
                for _ in range(int(close_iters)):
                    barrier = (_conv3d_mask(barrier, kernel, padding=1) > 0).float()
                    barrier = (_conv3d_mask(barrier, kernel, padding=1) == 27).float()
            barrier = (barrier[0, 0] > 0)
            free = ~barrier
            outside = _flood_fill_outside(free)
            interior = (~outside)

            # Step 2: optional signed constraint — keep where SDF is negative
            solid = (sdf_grid < 0)
            occ_mask = interior & solid

            # Keep only the largest connected component to remove stray islands.
            # occ_mask = _largest_connected_component(occ_mask, connectivity=6)

            # Step 3: near-surface band
            if surface_only:
                # Use unsigned band for robustness; |sdf| band could drop details if signs flip
                band = float(surface_band) if surface_band is not None else max(float(sdf_band), br_range * vx)
                near = dist_grid <= band
                occ_mask = occ_mask & near

        # Flatten + index
        occupy_mask = occ_mask.reshape(-1)
        occupy_xyz = self.voxel_xyz[occupy_mask]


        # Save for later use
        self.occupy_xyz = occupy_xyz.float()

        if return_sdf:
            if fill_mode == "unsigned":
                sdf_grid = torch.from_numpy(dist_all.reshape(Nx, Ny, Nz)).to(device=self.voxel_xyz.device, dtype=torch.float32)
            else:
                sdf_grid = torch.from_numpy(sdf_all.reshape(Nx, Ny, Nz)).to(device=self.voxel_xyz.device, dtype=torch.float32)
            return occupy_xyz, sdf_grid
        return occupy_xyz
    
    def init_from_meshs(self, meshes, surface_only=True):
        occupy_xyz_list = []
        mem_id = []
        if isinstance(self.base_voxel_size, list):
            mem_voxel_size_list = self.base_voxel_size
            is_list_voxel_size = True
        else:
            is_list_voxel_size = False
        for i, mesh in enumerate(meshes):
            if is_list_voxel_size:
                self.base_voxel_size = mem_voxel_size_list[i]
            occupy_xyz = self.init_from_mesh(mesh, surface_only=surface_only)
            occupy_xyz_list.append(occupy_xyz)
            mem_id.append(torch.ones((occupy_xyz.shape[0]), dtype=torch.int64, device=occupy_xyz.device) * i)
        occupy_xyz = torch.cat(occupy_xyz_list, dim=0)
        mem_id = torch.cat(mem_id, dim=0)
        self.occupy_xyz = occupy_xyz.float()
        self.mem_id = mem_id
        if is_list_voxel_size:
            self.base_voxel_size = mem_voxel_size_list
        return occupy_xyz

def get_mesh_barrier(mesh, base_voxel_size, padding=0.0, relative_padding=False, chunk_size=2000000
                       , sdf_band=0.0, close_iters=0, surface_only=True):
        
        # 1) Convert to Open3D legacy mesh
        o3_mesh = _mesh_to_o3d_legacy(mesh)

        # 2) Bounds with optional padding
        aabb = o3_mesh.get_axis_aligned_bounding_box()
        # Make explicit writable copies; Open3D can return read-only buffers
        bmin = np.array(aabb.min_bound, dtype=np.float64, copy=True)
        bmax = np.array(aabb.max_bound, dtype=np.float64, copy=True)
        if relative_padding:
            pad = (bmax - bmin) * float(padding)
            bmin -= pad
            bmax += pad
        else:
            p = float(padding)
            bmin -= p
            bmax += p

        # 3) Convert world bounds -> integer grid bounds (aligned to base_voxel_size)
        vx = float(base_voxel_size)
        x_grid_min = int(np.floor(bmin[0] / vx))
        y_grid_min = int(np.floor(bmin[1] / vx))
        z_grid_min = int(np.floor(bmin[2] / vx))
        x_grid_max = int(np.ceil (bmax[0] / vx))
        y_grid_max = int(np.ceil (bmax[1] / vx))
        z_grid_max = int(np.ceil (bmax[2] / vx))

        Nx = x_grid_max - x_grid_min
        Ny = y_grid_max - y_grid_min
        Nz = z_grid_max - z_grid_min
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError("Invalid grid dimensions computed from mesh bounds.")

        voxel_xyz = _build_voxel_xyz(
            x_grid_min,
            x_grid_max,
            y_grid_min,
            y_grid_max,
            z_grid_min,
            z_grid_max,
            vx,
        )


        # 5) Build Open3D RaycastingScene
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        # 6) Compute (signed and/or unsigned) distance at voxel centers in chunks
        pts_np = voxel_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        N = pts_np.shape[0]
        # Always prepare arrays; some modes use both (hybrid logic for robustness)
        dist_all = np.empty((N,), dtype=np.float32)


        chunk_size = int(chunk_size)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            pts_chunk = o3d.core.Tensor(pts_np[start:end], dtype=o3d.core.Dtype.Float32)
            d = scene.compute_distance(pts_chunk).numpy()
            dist_all[start:end] = d

        # 7) Occupancy mask from unsigned or signed distance, with optional closing
        # Barrier from unsigned distance <= tau, then outside flood-fill
        tau = max(float(sdf_band), 0.75 * vx)  # thicker barrier for robustness
        dist_grid = torch.from_numpy(dist_all.reshape(Nx, Ny, Nz)).to(device=voxel_xyz.device, dtype=torch.float32)
        # Optional closing on the barrier to seal tiny cracks
        barrier = (dist_grid <= tau).float().unsqueeze(0).unsqueeze(0)
        if int(close_iters) > 0:
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=barrier.device)
            for _ in range(int(close_iters)):
                barrier = (_conv3d_mask(barrier, kernel, padding=1) > 0).float()
                barrier = (_conv3d_mask(barrier, kernel, padding=1) == 27).float()
        barrier = (barrier[0,0] > 0)
        free = ~barrier
        outside = _flood_fill_outside(free)
        occ_mask = (~outside)
        
        if surface_only:
            band = max(float(sdf_band), 0.75 * vx)
            near = dist_grid <= band
            occ_mask = occ_mask & near
        
        return voxel_xyz[occ_mask.reshape(-1)]

def get_mesh_barrier_in_out(mesh, base_voxel_size, padding=0.0, relative_padding=False, chunk_size=2000000
                       , sdf_band=0.0, close_iters=0):
        
        # 1) Convert to Open3D legacy mesh
        o3_mesh = _mesh_to_o3d_legacy(mesh)

        # 2) Bounds with optional padding
        aabb = o3_mesh.get_axis_aligned_bounding_box()
        # Make explicit writable copies; Open3D can return read-only buffers
        bmin = np.array(aabb.min_bound, dtype=np.float64, copy=True) 
        bmax = np.array(aabb.max_bound, dtype=np.float64, copy=True) * 1.2
        if relative_padding:
            pad = (bmax - bmin) * float(padding)
            bmin -= pad
            bmax += pad
        else:
            p = float(padding)
            bmin -= p
            bmax += p

        # 3) Convert world bounds -> integer grid bounds (aligned to base_voxel_size)
        vx = float(base_voxel_size)
        x_grid_min = int(np.floor(bmin[0] / vx))
        y_grid_min = int(np.floor(bmin[1] / vx))
        z_grid_min = int(np.floor(bmin[2] / vx))
        x_grid_max = int(np.ceil (bmax[0] / vx))
        y_grid_max = int(np.ceil (bmax[1] / vx))
        z_grid_max = int(np.ceil (bmax[2] / vx))

        Nx = x_grid_max - x_grid_min
        Ny = y_grid_max - y_grid_min
        Nz = z_grid_max - z_grid_min
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError("Invalid grid dimensions computed from mesh bounds.")

        voxel_xyz = _build_voxel_xyz(
            x_grid_min,
            x_grid_max,
            y_grid_min,
            y_grid_max,
            z_grid_min,
            z_grid_max,
            vx,
        )


        # 5) Build Open3D RaycastingScene
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(o3_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)

        # 6) Compute (signed and/or unsigned) distance at voxel centers in chunks
        pts_np = voxel_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        N = pts_np.shape[0]
        # Always prepare arrays; some modes use both (hybrid logic for robustness)
        dist_all = np.empty((N,), dtype=np.float32)


        chunk_size = int(chunk_size)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            pts_chunk = o3d.core.Tensor(pts_np[start:end], dtype=o3d.core.Dtype.Float32)
            d = scene.compute_distance(pts_chunk).numpy()
            dist_all[start:end] = d

        # 7) Occupancy mask from unsigned or signed distance, with optional closing
        # Barrier from unsigned distance <= tau, then outside flood-fill
        tau = max(float(sdf_band), 0.5 * vx)  # thicker barrier for robustness
        dist_grid = torch.from_numpy(dist_all.reshape(Nx, Ny, Nz)).to(device=voxel_xyz.device, dtype=torch.float32)
        # Optional closing on the barrier to seal tiny cracks
        barrier = (dist_grid <= tau).float().unsqueeze(0).unsqueeze(0)
        if int(close_iters) > 0:
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=barrier.device)
            for _ in range(int(close_iters)):
                barrier = (_conv3d_mask(barrier, kernel, padding=1) > 0).float()
                barrier = (_conv3d_mask(barrier, kernel, padding=1) == 27).float()
        barrier = (barrier[0,0] > 0)
        free = ~barrier
        outside = _flood_fill_outside(free)
        occ_mask = (~outside)
        
        band = max(float(sdf_band), 1.5 * vx)
        near = dist_grid <= band
        
        inside_xyzs = voxel_xyz[(occ_mask & near).reshape(-1)]
        outside_xyzs = voxel_xyz[(~occ_mask & near).reshape(-1)]
        
        return inside_xyzs, outside_xyzs
