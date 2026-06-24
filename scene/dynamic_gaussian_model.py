#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from utils.general_utils import get_expon_lr_func, build_scaling_rotation
from torch import nn
from scene.gaussian_model import GaussianModel
from pytorch3d.transforms import quaternion_to_matrix, quaternion_multiply
import fpsample
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import pytorch3d
import math
from scene.voxel_grid import VoxelGrid
from utils.training_utils import create_lr_scheduler
from utils.phys_loss_utils import sample_points_on_mesh

def sample_gs_pdf(x: torch.Tensor,
                    mean: torch.Tensor,
                    scaling: torch.Tensor,
                    rotaion: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x    : (..., D)          – query vector(s)
    mean : (..., D)          – matching means
    cov  : (..., D, D)       – matching full cov. matrices
                               (must be symmetric positive-definite)

    Returns
    -------
    log p(x)  : (...)        – log-density for every leading batch index
    """
    L = build_scaling_rotation(scaling, rotaion)  # (..., D, D)

    # 2. Solve  L y = (x-μ)   ⇒   y = L⁻¹ (x-μ)
    xm = x - mean                                    # (..., D)
    d0 = L[..., 0, 0]                      # (...,)
    d1 = L[..., 1, 1]
    d2 = L[..., 2, 2]

    # forward substitution --------------------------------------------
    y0 =  xm[..., 0] / d0

    y1 = (xm[..., 1] - L[..., 1, 0] * y0) / d1

    y2 = (xm[..., 2]
        - L[..., 2, 0] * y0
        - L[..., 2, 1] * y1) / d2

    y  = torch.stack((y0, y1, y2), dim=-1)   # (..., 3)

    # 3. Mahalanobis term  (x-μ)ᵀ Σ⁻¹ (x-μ)  =  ‖y‖²
    maha = (y ** 2).sum(-1)                          # (...)
    return torch.exp(-0.5 * maha)

def sample_gs_pdf_warp(x: torch.Tensor,
                    mean: torch.Tensor,
                    scaling: torch.Tensor,
                    rotaion: torch.Tensor) -> torch.Tensor:
    N, K = x.shape[0], x.shape[1]
    res = sample_gs_pdf(
        x.reshape(N * K, 3),
        mean.reshape(N * K, 3),
        scaling.reshape(N * K, 3),
        rotaion.reshape(N * K, 4),
    )
    res = res.reshape(N, K)
    return res

def distCUDA2(points, k=3):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=k)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

def lowbit(x):
    return x & -x

class BitMotionBase:

    def __init__(self, total_time, cp_num):
        self.total_time = total_time
        self.cp_num = cp_num
        self.rot_activate = torch.nn.functional.normalize
        cp_deform = torch.zeros((total_time+1, cp_num, 7), device="cuda", dtype=torch.float32)
        self.cp_deform = nn.Parameter(cp_deform.requires_grad_(True))
        self.converted_to_independent = False
    
    def convert_to_independent(self):
        if self.converted_to_independent:
            return
        new_deform = torch.zeros_like(self.cp_deform)
        for i in range(self.total_time):
            cur_deform = self.query_no_activate(i)
            new_deform[i+1] = cur_deform
        self.cp_deform = nn.Parameter(new_deform.requires_grad_(True))
        self.converted_to_independent = True
    
    def query(self, time):
        final_deform = torch.zeros((self.cp_num, 7), device="cuda", dtype=torch.float32)
        query_time = time
        time = time + 1
        if hasattr(self, "converted_to_independent") and self.converted_to_independent:
            final_deform = self.cp_deform[time]
            final_rot = final_deform[:, :4]
            final_trans = final_deform[:, 4:]
            final_rot = self.rot_activate(final_rot)
            if query_time == 0:
                final_rot = final_rot.detach()
                final_trans = final_trans.detach()
            return final_rot, final_trans
        else:
            while(time > 0):
                final_deform = final_deform + self.cp_deform[time]
                time = time - lowbit(time)
            final_rot = final_deform[:, :4]
            final_trans = final_deform[:, 4:]
            final_rot[:,0] += 1.0
            final_rot = self.rot_activate(final_rot)
            if query_time == 0:
                final_rot = final_rot.detach()
                final_trans = final_trans.detach()
            return final_rot, final_trans
    
    def query_whole(self, max_time):
        total_rot, total_trans = [], []
        for t in range(max_time):
            cur_rot, cur_trans = self.query(t)
            total_rot.append(cur_rot.unsqueeze(0))
            total_trans.append(cur_trans.unsqueeze(0))
        total_rot = torch.cat(total_rot, dim=0)
        total_trans = torch.cat(total_trans, dim=0)
        return total_rot, total_trans

    def query_range(self, start_time, end_time):
        total_rot, total_trans = [], []
        for t in range(start_time, end_time):
            cur_rot, cur_trans = self.query(t)
            total_rot.append(cur_rot.unsqueeze(0))
            total_trans.append(cur_trans.unsqueeze(0))
        total_rot = torch.cat(total_rot, dim=0)
        total_trans = torch.cat(total_trans, dim=0)
        return total_rot, total_trans
    
    def query_no_activate(self, time):
        final_deform = torch.zeros((self.cp_num, 7), device="cuda", dtype=torch.float32)
        time = time + 1
        if hasattr(self, "converted_to_independent") and self.converted_to_independent:
            final_deform = self.cp_deform[time]
            return final_deform
        else:
            while(time > 0):
                final_deform = final_deform + self.cp_deform[time]
                time = time - lowbit(time)
        final_deform[:,0] += 1.0
        return final_deform
    

class DynamicGaussianModel:



    def __init__(self, gaussians: GaussianModel, total_time: int, spatial_lr_scale: float = 1.0, n_cp_num: int = 3):
        gaussians.detach_all()
        self.gaussians = gaussians
        self.total_time = total_time
        self.cp_radius = torch.empty(0).cuda()
        self.cp_rotation = torch.empty(0).cuda()
        self.cp_center = torch.empty(0).cuda()
        self.spatial_lr_scale = spatial_lr_scale
        self.n_cp_num = n_cp_num
        self.c_cp_deform = None
        self.c_cp_radius = None
        self.c_cp_center = None
        self.c_cp_rotation = None
        self.mult_rot_way = 0
        
    def detach_all(self):
        self.cp_center = self.cp_center.detach()
        self.cp_deform.cp_deform = self.cp_deform.cp_deform.detach()
        self.cp_rotation = self.cp_rotation.detach()
        self.cp_radius = self.cp_radius.detach()
        self.c_cp_center = self.c_cp_center.detach() if self.c_cp_center is not None else None
        self.c_cp_deform.cp_deform = self.c_cp_deform.cp_deform.detach() if self.c_cp_deform is not None else None
        self.c_cp_rotation = self.c_cp_rotation.detach() if self.c_cp_rotation is not None else None
        self.c_cp_radius = self.c_cp_radius.detach() if self.c_cp_radius is not None else None
    
    def _cluster_control_points(self, pc, init_cp_num):
        fps_samples_idx = fpsample.fps_sampling(pc, init_cp_num)
        init_xyz = pc[fps_samples_idx]
        kmeans = KMeans(n_clusters=init_cp_num, init=init_xyz, n_init=1)
        kmeans.fit(pc)
        return torch.from_numpy(kmeans.cluster_centers_).cuda()

    def _init_control_point_tensors(self, cp_xyz):
        init_cp_num = cp_xyz.shape[0]
        self.cp_center = cp_xyz.detach().clone()
        cp_radius = torch.sqrt(distCUDA2(cp_xyz, k=3)).unsqueeze(-1).repeat(1, 3)
        cp_radius = torch.log(cp_radius)
        self.cp_radius = nn.Parameter(cp_radius.requires_grad_(True))
        cp_rotation = torch.zeros((cp_xyz.shape[0], 4), device="cuda", dtype=torch.float32)
        cp_rotation[:, 0] = 1.0
        self.cp_rotation = nn.Parameter(cp_rotation.requires_grad_(True))

        padded_total_time = 2 ** math.ceil(math.log2(self.total_time))
        self.cp_deform = BitMotionBase(padded_total_time, init_cp_num)

    def init_control_points_from_gaussians(self, init_cp_num, br_range=0.75):
        self.br_range = br_range
        pc = self.gaussians._xyz.detach().cpu().numpy()
        cp_xyz = self._cluster_control_points(pc, init_cp_num)
        self._init_control_point_tensors(cp_xyz)
        self.voxel_xyz = self.gaussians._xyz.detach().clone()
        self.voxel_grid = None
        self.mem_init_voxel_size = None

    def init_control_points(self, init_cp_num, init_mesh, init_voxel_size=0.015
                            , br_range=0.75, cp_surface_only=False, target_num=8000):
        self.br_range = br_range

        voxel_grid = VoxelGrid(init_voxel_size, 500, 0.001)
        pc_tensor = voxel_grid.init_from_mesh(init_mesh, padding=0.0, sdf_band=0.0, close_iters=0
                                       , surface_only=False, br_range=br_range).float()
        self.voxel_xyz = pc_tensor
        self.voxel_grid = voxel_grid

        pc = pc_tensor.detach().cpu().numpy()
        cp_xyz = self._cluster_control_points(pc, init_cp_num)

        if cp_surface_only:
            cp_xyz = sample_points_on_mesh(init_mesh, init_cp_num)

        l = init_voxel_size/10.0
        r = init_voxel_size*10.0
        result = init_voxel_size
        eps = 1e-6
        while abs(r-l)>eps:
            mid = (l+r)/2.0
            voxel_grid = VoxelGrid(mid, 500, 0.001)
            pc_tensor = voxel_grid.init_from_mesh(init_mesh, surface_only=True, br_range=br_range).float()
            if pc_tensor.shape[0] > target_num:
                l = mid
            else:
                r = mid
                result = mid
        init_voxel_size = result
        voxel_grid = VoxelGrid(init_voxel_size, 500, 0.001)
        self.voxel_xyz = voxel_grid.init_from_mesh(init_mesh, surface_only=True, br_range=br_range).float()

        _, valid_mask1 = self.prune_adundant_cp(cp_xyz, self.gaussians._xyz.detach())
        _, valid_mask2 = self.prune_adundant_cp(cp_xyz, self.voxel_xyz)
        valid_mask = valid_mask1 & valid_mask2
        cp_xyz = cp_xyz[valid_mask]
        self._init_control_point_tensors(cp_xyz)
        self.voxel_grid = voxel_grid
        self.mem_init_voxel_size = init_voxel_size
    
    def cal_nn_weight(self, x: torch.Tensor, detach_node_radius=False):
        K = self.n_cp_num
        nodes = self.cp_center[..., :3].detach()

        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)
        nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]
        if detach_node_radius:
            nn_radius = self.gaussians.scaling_activation(self.cp_radius[nn_idxs].detach())
            nn_rotation = self.cp_rotation[nn_idxs].detach()
        else:
            nn_radius = self.gaussians.scaling_activation(self.cp_radius[nn_idxs])
            nn_rotation = self.cp_rotation[nn_idxs]
        nn_means = nodes[nn_idxs]
        nn_xyz = x.unsqueeze(1).repeat(1, K, 1)
        nn_weight = sample_gs_pdf_warp(nn_xyz, nn_means, nn_radius, nn_rotation)
        nn_weight = nn_weight + 1e-7
        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)
        return nn_weight, nn_dist, nn_idxs
        
    def cal_nn_weight_additional(self, x: torch.Tensor, detach_node_radius=False):
        K = self.n_cp_num
        nodes = self.c_cp_center[..., :3].detach()

        nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)
        nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]
        if detach_node_radius:
            nn_radius = self.gaussians.scaling_activation(self.c_cp_radius[nn_idxs].detach())
            nn_rotation = self.c_cp_rotation[nn_idxs].detach()
        else:
            nn_radius = self.gaussians.scaling_activation(self.c_cp_radius[nn_idxs])
            nn_rotation = self.c_cp_rotation[nn_idxs]
        nn_means = nodes[nn_idxs]
        nn_xyz = x.unsqueeze(1).repeat(1, K, 1)
        nn_weight = sample_gs_pdf_warp(nn_xyz, nn_means, nn_radius, nn_rotation)
        nn_weight = nn_weight + 1e-7
        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)
        return nn_weight, nn_dist, nn_idxs
    
    def get_scaling(self, time = None):
        return self.gaussians.get_scaling
    
    def get_xyz_rotation_additional(self, time, detach_node_radius=True):
        static_rotation = self.gaussians.get_rotation.detach()
        static_xyz = self.gaussians.get_xyz.detach()
        nn_weight, _, nn_idx = self.cal_nn_weight_additional(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.c_cp_deform.query(time)
        rotation = (cp_rot[nn_idx] * nn_weight[..., None]).sum(dim=1)
        final_rotation = rotation
        nn_cp = self.c_cp_center[nn_idx,...,:3].detach()
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], static_xyz[:, None]-nn_cp) + nn_cp + cp_trans[nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
        final_xyz = Ax_avg
        delta_xyz = final_xyz - static_xyz
        if time == 0:
            delta_xyz = delta_xyz.detach()
            final_rotation = final_rotation.detach()
        
        return delta_xyz, final_rotation
    
    def get_xyz_rotation(self, time, detach_node_radius=True):
        static_rotation = self.gaussians.get_rotation.detach()
        static_xyz = self.gaussians.get_xyz.detach()
        nn_weight, _, nn_idx = self.cal_nn_weight(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.cp_deform.query(time)
        rotation = (cp_rot[nn_idx] * nn_weight[..., None]).sum(dim=1)
        final_rotation = quaternion_multiply(rotation, static_rotation)
        nn_cp = self.cp_center[nn_idx,...,:3].detach()
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], static_xyz[:, None]-nn_cp) + nn_cp + cp_trans[nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
        final_xyz = Ax_avg
        
        if self.c_cp_deform is not None:
            if self.detach_first_layer:
                final_xyz = final_xyz.detach()
                final_rotation = final_rotation.detach()
                rotation = rotation.detach()
            delta_xyz, delta_rotation = self.get_xyz_rotation_additional(time, detach_node_radius)
            final_xyz = final_xyz + delta_xyz
            if self.mult_rot_way == 0:
                final_rotation = quaternion_multiply(delta_rotation, final_rotation)
            elif self.mult_rot_way == 1:
                cur_rotation = rotation + delta_rotation
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
            elif self.mult_rot_way == 2:
                cur_rotation = quaternion_multiply(rotation, delta_rotation)
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
        if time == 0:
            final_xyz = final_xyz.detach()
            final_rotation = final_rotation.detach()
        
        return final_xyz, final_rotation

    def query_xyz_time_additional(self, xyz, time, detach_node_radius = True):
        static_xyz = xyz
        nn_weight, _, nn_idx = self.cal_nn_weight_additional(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.c_cp_deform.query(time)
        nn_cp = self.c_cp_center[nn_idx,...,:3].detach()
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], static_xyz[:, None]-nn_cp) + nn_cp + cp_trans[nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
        final_xyz = Ax_avg
        delta_xyz = final_xyz - static_xyz
        if time == 0:
            delta_xyz = delta_xyz.detach()
        return delta_xyz
    
    def query_xyz_time(self, xyz, time, detach_node_radius = True):
        static_xyz = xyz
        nn_weight, _, nn_idx = self.cal_nn_weight(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.cp_deform.query(time)
        nn_cp = self.cp_center[nn_idx,...,:3].detach()
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], static_xyz[:, None]-nn_cp) + nn_cp + cp_trans[nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
        final_xyz = Ax_avg
        if self.c_cp_deform is not None:
            if self.detach_first_layer:
                final_xyz = final_xyz.detach()
            delta_xyz = self.query_xyz_time_additional(xyz, time, detach_node_radius)
            final_xyz = final_xyz + delta_xyz
        if time == 0:
            final_xyz = final_xyz.detach()
        return final_xyz
    
    def query_xyz(self, xyz):
        final_xyz = []
        for i in range(self.total_time):
            cur_xyz = self.query_xyz_time(xyz, i)
            final_xyz.append(cur_xyz)
        final_xyz = torch.stack(final_xyz, dim=0)
        return final_xyz
    
    def get_xyz_rotation_whole(self, detach_node_radius=True):
        static_rotation = self.gaussians.get_rotation.detach()
        static_xyz = self.gaussians.get_xyz.detach()
        
        nn_weight, _, nn_idx = self.cal_nn_weight(static_xyz, detach_node_radius)
        
        cp_rot, cp_trans = self.cp_deform.query_whole(self.total_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(self.total_time, 1, 1)
        rotation = (cp_rot[:, nn_idx] * nn_weight[..., None]).sum(dim=2)
        final_rotation = quaternion_multiply(rotation, static_rotation.unsqueeze(0).repeat(self.total_time, 1, 1))
        final_rotation = self.gaussians.rotation_activation(final_rotation, dim=-1)
        nn_cp = self.cp_center[nn_idx,...,:3].detach()
        nn_cp = nn_cp.unsqueeze(0).repeat(self.total_time, 1, 1, 1)
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('tnkab,tnkb->tnka', local_rot_matrix[:,nn_idx], static_xyz[:, None].unsqueeze(0).repeat(self.total_time, 1, 1, 1)-nn_cp) + nn_cp + cp_trans[:, nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        if self.c_cp_deform is not None:
            if self.detach_first_layer:
                final_xyz = final_xyz.detach()
                final_rotation = final_rotation.detach()
                rotation = rotation.detach()
            delta_xyz, delta_rotation = self.get_xyz_rotation_additional_whole(detach_node_radius)
            final_xyz = final_xyz + delta_xyz
            if self.mult_rot_way == 0:
                final_rotation = quaternion_multiply(delta_rotation, final_rotation)
            elif self.mult_rot_way == 1:
                cur_rotation = rotation + delta_rotation
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
            elif self.mult_rot_way == 2:
                cur_rotation = quaternion_multiply(rotation, delta_rotation)
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
        final_rotation = torch.cat([final_rotation[0:1].detach(), final_rotation[1:]], dim=0)
        final_xyz = torch.cat([final_xyz[0:1].detach(), final_xyz[1:]], dim=0)
            
        return final_xyz, final_rotation

    def get_xyz_rotation_range(self, start_time, end_time, detach_node_radius=True):
        chunk_time = end_time - start_time
        static_rotation = self.gaussians.get_rotation.detach()
        static_xyz = self.gaussians.get_xyz.detach()

        nn_weight, _, nn_idx = self.cal_nn_weight(static_xyz, detach_node_radius)

        cp_rot, cp_trans = self.cp_deform.query_range(start_time, end_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(chunk_time, 1, 1)
        rotation = (cp_rot[:, nn_idx] * nn_weight[..., None]).sum(dim=2)
        final_rotation = quaternion_multiply(
            rotation,
            static_rotation.unsqueeze(0).repeat(chunk_time, 1, 1),
        )
        final_rotation = self.gaussians.rotation_activation(final_rotation, dim=-1)
        nn_cp = self.cp_center[nn_idx, ..., :3].detach()
        nn_cp = nn_cp.unsqueeze(0).repeat(chunk_time, 1, 1, 1)
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = (
            torch.einsum(
                'tnkab,tnkb->tnka',
                local_rot_matrix[:, nn_idx],
                static_xyz[:, None].unsqueeze(0).repeat(chunk_time, 1, 1, 1) - nn_cp,
            )
            + nn_cp
            + cp_trans[:, nn_idx]
        )
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        if self.c_cp_deform is not None:
            if self.detach_first_layer:
                final_xyz = final_xyz.detach()
                final_rotation = final_rotation.detach()
                rotation = rotation.detach()
            delta_xyz, delta_rotation = self.get_xyz_rotation_additional_range(
                start_time,
                end_time,
                detach_node_radius,
            )
            final_xyz = final_xyz + delta_xyz
            if self.mult_rot_way == 0:
                final_rotation = quaternion_multiply(delta_rotation, final_rotation)
            elif self.mult_rot_way == 1:
                cur_rotation = rotation + delta_rotation
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
            elif self.mult_rot_way == 2:
                cur_rotation = quaternion_multiply(rotation, delta_rotation)
                final_rotation = quaternion_multiply(cur_rotation, static_rotation)
        if start_time == 0:
            final_rotation = torch.cat([final_rotation[0:1].detach(), final_rotation[1:]], dim=0)
            final_xyz = torch.cat([final_xyz[0:1].detach(), final_xyz[1:]], dim=0)

        return final_xyz, final_rotation
    
    def get_xyz_rotation_additional_whole(self, detach_node_radius = True):
        static_xyz = self.gaussians.get_xyz.detach()
        
        nn_weight, _, nn_idx = self.cal_nn_weight_additional(static_xyz, detach_node_radius)
        
        cp_rot, cp_trans = self.c_cp_deform.query_whole(self.total_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(self.total_time, 1, 1)
        rotation = (cp_rot[:, nn_idx] * nn_weight[..., None]).sum(dim=2)
        final_rotation = rotation
        nn_cp = self.c_cp_center[nn_idx,...,:3].detach()
        nn_cp = nn_cp.unsqueeze(0).repeat(self.total_time, 1, 1, 1)
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('tnkab,tnkb->tnka', local_rot_matrix[:,nn_idx], static_xyz[:, None].unsqueeze(0).repeat(self.total_time, 1, 1, 1)-nn_cp) + nn_cp + cp_trans[:, nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        delta_xyz = final_xyz - static_xyz.unsqueeze(0).repeat(self.total_time, 1, 1)

        final_rotation = torch.cat([final_rotation[0:1].detach(), final_rotation[1:]], dim=0)
        delta_xyz = torch.cat([delta_xyz[0:1].detach(), delta_xyz[1:]], dim=0)
        
        return delta_xyz, final_rotation

    def get_xyz_rotation_additional_range(self, start_time, end_time, detach_node_radius=True):
        chunk_time = end_time - start_time
        static_xyz = self.gaussians.get_xyz.detach()

        nn_weight, _, nn_idx = self.cal_nn_weight_additional(static_xyz, detach_node_radius)

        cp_rot, cp_trans = self.c_cp_deform.query_range(start_time, end_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(chunk_time, 1, 1)
        rotation = (cp_rot[:, nn_idx] * nn_weight[..., None]).sum(dim=2)
        final_rotation = rotation
        nn_cp = self.c_cp_center[nn_idx, ..., :3].detach()
        nn_cp = nn_cp.unsqueeze(0).repeat(chunk_time, 1, 1, 1)
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = (
            torch.einsum(
                'tnkab,tnkb->tnka',
                local_rot_matrix[:, nn_idx],
                static_xyz[:, None].unsqueeze(0).repeat(chunk_time, 1, 1, 1) - nn_cp,
            )
            + nn_cp
            + cp_trans[:, nn_idx]
        )
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        delta_xyz = final_xyz - static_xyz.unsqueeze(0).repeat(chunk_time, 1, 1)

        if start_time == 0:
            final_rotation = torch.cat([final_rotation[0:1].detach(), final_rotation[1:]], dim=0)
            delta_xyz = torch.cat([delta_xyz[0:1].detach(), delta_xyz[1:]], dim=0)

        return delta_xyz, final_rotation
    
    def query_xyz_time_whole(self, xyz, detach_node_radius = True):
        static_xyz = xyz
        nn_weight, _, nn_idx = self.cal_nn_weight(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.cp_deform.query_whole(self.total_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(self.total_time, 1, 1)
        nn_cp = self.cp_center[nn_idx,...,:3].detach()
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('tnkab,tnkb->tnka', local_rot_matrix[:,nn_idx]
                          , static_xyz[:, None].unsqueeze(0).repeat(self.total_time, 1, 1, 1)-nn_cp) + nn_cp + cp_trans[:, nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        if self.c_cp_deform is not None:
            if self.detach_first_layer:
                final_xyz = final_xyz.detach()
            delta_xyz = self.query_xyz_time_additional_whole(xyz, detach_node_radius)
            final_xyz = final_xyz + delta_xyz
        final_xyz = torch.cat([final_xyz[0:1].detach(), final_xyz[1:]], dim=0)
        
        return final_xyz
    
    def query_xyz_time_additional_whole(self, xyz, detach_node_radius = True):
        static_xyz = xyz
        nn_weight, _, nn_idx = self.cal_nn_weight_additional(static_xyz, detach_node_radius)
        cp_rot, cp_trans = self.c_cp_deform.query_whole(self.total_time)
        nn_weight = nn_weight.unsqueeze(0).repeat(self.total_time, 1, 1)
        nn_cp = self.c_cp_center[nn_idx,...,:3].detach()
            
        local_rot_matrix = quaternion_to_matrix(cp_rot)
        Ax = torch.einsum('tnkab,tnkb->tnka', local_rot_matrix[:,nn_idx]
                          , static_xyz[:, None].unsqueeze(0).repeat(self.total_time, 1, 1, 1)-nn_cp) + nn_cp + cp_trans[:, nn_idx]
        Ax_avg = (Ax * nn_weight[..., None]).sum(dim=2)
        final_xyz = Ax_avg
        delta_xyz = final_xyz - static_xyz.unsqueeze(0).repeat(self.total_time, 1, 1)
        delta_xyz = torch.cat([delta_xyz[0:1].detach(), delta_xyz[1:]], dim=0)
        return delta_xyz
    
    def get_cp_rotation(self, time):
        cp_rotation = self.gaussians.rotation_activation(self.cp_rotation)
        deform_rotation, _ = self.cp_deform.query(time)
        final_rotation = quaternion_multiply(deform_rotation, cp_rotation)
        return final_rotation
    def get_cp_scaling(self):
        cp_scaling = self.gaussians.scaling_activation(self.cp_radius)
        return cp_scaling
    def get_cp_position(self, time):
        _, cp_trans = self.cp_deform.query(time)
        return self.cp_center+cp_trans
    
    def get_additional_cp_rotation(self, time):
        cp_rotation = self.gaussians.rotation_activation(self.c_cp_rotation)
        deform_rotation, _ = self.c_cp_deform.query(time)
        final_rotation = quaternion_multiply(deform_rotation, cp_rotation)
        return final_rotation
    def get_additional_cp_scaling(self):
        cp_scaling = self.gaussians.scaling_activation(self.c_cp_radius)
        return cp_scaling
    def get_additional_cp_position(self, time):
        _, cp_trans = self.c_cp_deform.query(time)
        return self.c_cp_center+cp_trans
    
    def convert_cp_to_independent(self):
        if self.cp_deform.converted_to_independent:
            return
        self.cp_deform.convert_to_independent()
        d = self.replace_tensor_to_optimizer(self.cp_deform.cp_deform.detach(), "cp_deform")
        self.cp_deform.cp_deform = d["cp_deform"]
        
    def convert_additional_cp_to_independent(self):
        if self.c_cp_deform.converted_to_independent:
            return
        self.c_cp_deform.convert_to_independent()
        d = self.replace_tensor_to_optimizer(self.c_cp_deform.cp_deform.detach(), "c_cp_deform")
        self.c_cp_deform.cp_deform = d["c_cp_deform"]
        self.cp_deform.cp_deform = self.cp_deform.cp_deform.detach()

    def get_features(self, time = None):
        return self.gaussians.get_features
    
    def get_opacity(self, time = None):
        return self.gaussians.get_opacity
    
    def prune_adundant_cp(self, cp_xyzs, ref_xyzs):
        """
        Keep only control points that appear in the top-K nearest control points
        of at least one reference point, where K is the deformation KNN count.
        `ref_xyzs` may be `[N, 3]`, `[T, N, 3]`, or a list/tuple of such tensors.
        """

        if not torch.is_tensor(cp_xyzs):
            cp_xyzs = torch.as_tensor(cp_xyzs, device="cuda", dtype=torch.float32)
        if cp_xyzs.ndim != 2 or cp_xyzs.shape[-1] != 3:
            raise ValueError(f"cp_xyzs must have shape [M, 3], got {tuple(cp_xyzs.shape)}")
        if cp_xyzs.shape[0] == 0:
            return cp_xyzs

        cp_xyzs = cp_xyzs.to(dtype=cp_xyzs.dtype, device=cp_xyzs.device)
        K = max(1, min(self.n_cp_num, cp_xyzs.shape[0]))
        keep_mask = torch.zeros((cp_xyzs.shape[0],), device=cp_xyzs.device, dtype=torch.bool)
        chunk_size = 65536
        saw_ref = False

        def _mark_used_cp(cur_ref_xyzs):
            nonlocal saw_ref
            cur_ref_xyzs = torch.as_tensor(cur_ref_xyzs, device=cp_xyzs.device, dtype=cp_xyzs.dtype)
            if cur_ref_xyzs.numel() == 0:
                return
            if cur_ref_xyzs.shape[-1] != 3:
                raise ValueError(f"ref_xyzs must have a last dimension of size 3, got {tuple(cur_ref_xyzs.shape)}")

            cur_ref_xyzs = cur_ref_xyzs.reshape(-1, 3)
            if cur_ref_xyzs.shape[0] == 0:
                return

            saw_ref = True
            for beg in range(0, cur_ref_xyzs.shape[0], chunk_size):
                ref_chunk = cur_ref_xyzs[beg:beg + chunk_size]
                _, nn_idxs, _ = pytorch3d.ops.knn_points(
                    ref_chunk[None], cp_xyzs[None], None, None, K=K
                )
                keep_mask[nn_idxs.reshape(-1)] = True

        if isinstance(ref_xyzs, (list, tuple)):
            for cur_ref_xyzs in ref_xyzs:
                _mark_used_cp(cur_ref_xyzs)
        else:
            _mark_used_cp(ref_xyzs)

        if not saw_ref:
            return cp_xyzs

        return cp_xyzs[keep_mask], keep_mask

    def _build_adaptive_surface_voxel_xyz(self, init_mesh, init_voxel_size, target_num):
        l = init_voxel_size / 10.0
        r = init_voxel_size * 10.0
        result = init_voxel_size
        eps = 1e-6
        target_num = max(1, int(target_num))

        while abs(r - l) > eps:
            mid = (l + r) / 2.0
            voxel_grid = VoxelGrid(mid, 500, 0.001)
            voxel_xyz = voxel_grid.init_from_mesh(
                init_mesh,
                surface_only=True,
                br_range=self.br_range,
            ).float()
            if voxel_xyz.shape[0] > target_num:
                l = mid
            else:
                r = mid
                result = mid

        voxel_grid = VoxelGrid(result, 500, 0.001)
        voxel_xyz = voxel_grid.init_from_mesh(
            init_mesh,
            surface_only=True,
            br_range=self.br_range,
        ).float()
        return voxel_xyz, result
    
    def init_additional_control_points(self, init_cp_num, detach_first_layer=False,
                            init_mesh=None, init_voxel_size=0.015,
                            mult_rot_way=0, target_num=8000):
        if self.c_cp_deform is not None:
            voxel_grid = VoxelGrid(init_voxel_size, 500, 0.001)
            voxel_xyz = voxel_grid.init_from_mesh(init_mesh, surface_only=True, br_range=self.br_range).float()
            self.c_cp_center = self.c_cp_center.detach()
            self.c_cp_radius = nn.Parameter(self.c_cp_radius.requires_grad_(True))
            self.c_cp_rotation = nn.Parameter(self.c_cp_rotation.requires_grad_(True))
            self.c_cp_deform.cp_deform = nn.Parameter(self.c_cp_deform.cp_deform.requires_grad_(True))
            return voxel_xyz
        
        self.detach_first_layer = detach_first_layer
        original_init_cp_num = int(init_cp_num)
        voxel_grid = VoxelGrid(init_voxel_size, 500, 0.001)
        pc_tensor = voxel_grid.init_from_mesh(init_mesh, padding=0.0, sdf_band=0.0, close_iters=0
                                       , surface_only=False, br_range=self.br_range).float()

        pc = pc_tensor.detach().cpu().numpy()
        cp_xyz = self._cluster_control_points(pc, init_cp_num)

        adaptive_target_num = max(
            1,
            int(target_num) * original_init_cp_num // max(1, int(self.cp_center.shape[0])),
        )
        voxel_xyz, adaptive_init_voxel_size = self._build_adaptive_surface_voxel_xyz(
            init_mesh,
            init_voxel_size,
            adaptive_target_num,
        )
        self.mult_rot_way = mult_rot_way
        
        _, valid_mask1 = self.prune_adundant_cp(cp_xyz, self.gaussians._xyz.detach())
        _, valid_mask2 = self.prune_adundant_cp(cp_xyz, voxel_xyz)
        valid_mask = valid_mask1 & valid_mask2
        cp_xyz = cp_xyz[valid_mask]
        
        init_cp_num = cp_xyz.shape[0]
        self.c_cp_center = cp_xyz.detach().clone()
        cp_radius = torch.sqrt(distCUDA2(cp_xyz, k=3)).unsqueeze(-1).repeat(1, 3)
        cp_radius = torch.log(cp_radius)
        self.c_cp_radius = nn.Parameter(cp_radius.requires_grad_(True))
        cp_rotation = torch.zeros((cp_xyz.shape[0], 4), device="cuda", dtype=torch.float32)
        cp_rotation[:, 0] = 1.0
        self.c_cp_rotation = nn.Parameter(cp_rotation.requires_grad_(True))
        
        padded_total_time = 2 ** math.ceil(math.log2(self.total_time))

        self.c_cp_deform = BitMotionBase(padded_total_time, init_cp_num)
        
        return voxel_xyz
        
    def append_additional_cp_to_optimizer(self, training_args):
        l = [
            {'params': [self.c_cp_deform.cp_deform], 'lr': training_args.c_deform_lr_init * self.spatial_lr_scale, "name": "c_cp_deform"},
            {'params': [self.c_cp_radius], 'lr': training_args.c_cp_radius_lr_init * self.spatial_lr_scale, "name": "c_cp_radius"},
            {'params': [self.c_cp_rotation], 'lr': training_args.c_cp_rotation_lr_init * self.spatial_lr_scale, "name": "c_cp_rotation"},
        ]
        for param_group in l:
            self.optimizer.add_param_group(param_group)
        if training_args.use_lr_scheduler:
            self.init_lr_scheduler(training_args)
        else:
            self.c_cp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.c_deform_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.c_deform_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.c_deform_lr_delay_mult,
                                                        max_steps=training_args.c_deform_lr_max_steps)
            self.c_cp_radius_scheduler_args = get_expon_lr_func(lr_init=training_args.c_cp_radius_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.c_cp_radius_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.c_cp_radius_lr_delay_mult,
                                                        max_steps=training_args.c_cp_radius_lr_max_steps)
            self.c_cp_rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.c_cp_rotation_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.c_cp_rotation_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.c_cp_rotation_lr_delay_mult,
                                                        max_steps=training_args.c_cp_rotation_lr_max_steps)

    def training_setup(self, training_args):

        l = [
            {'params': [self.cp_radius], 'lr': training_args.cp_radius_lr_init * self.spatial_lr_scale, "name": "cp_radius"},
            {'params': [self.cp_rotation], 'lr': training_args.cp_rotation_lr_init * self.spatial_lr_scale, "name": "cp_rotation"},
            {'params': [self.cp_deform.cp_deform], 'lr': training_args.deform_lr_init * self.spatial_lr_scale, "name": "cp_deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, betas=(0.9, 0.95), eps=1e-15)
        self.use_lr_scheduler = training_args.use_lr_scheduler
        if training_args.use_lr_scheduler:
            self.init_lr_scheduler(training_args)
        else:
            self.cp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.deform_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.deform_lr_delay_mult,
                                                        max_steps=training_args.deform_lr_max_steps)
            self.cp_radius_scheduler_args = get_expon_lr_func(lr_init=training_args.cp_radius_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.cp_radius_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.cp_radius_lr_delay_mult,
                                                        max_steps=training_args.cp_radius_lr_max_steps)
            self.cp_rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.cp_rotation_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.cp_rotation_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.cp_rotation_lr_delay_mult,
                                                        max_steps=training_args.cp_rotation_lr_max_steps)
    
    def init_lr_scheduler(self, opt):
        lr_scheduler = create_lr_scheduler(
            self.optimizer,
            opt.iterations,
            opt.warmup_steps,
            scheduler_type=opt.scheduler_type,
        )
        self.lr_scheduler = lr_scheduler

    def update_learning_rate(self, iteration):
        if self.use_lr_scheduler:
            self.lr_scheduler.step(iteration)
            return
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if "cp_deform" in param_group["name"]:
                lr = self.cp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "cp_radius":
                lr = self.cp_radius_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "cp_rotation":
                lr = self.cp_rotation_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "c_cp_deform":
                lr = self.c_cp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "c_cp_radius":
                lr = self.c_cp_radius_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "c_cp_rotation":
                lr = self.c_cp_rotation_scheduler_args(iteration)
                param_group['lr'] = lr
        
    def save_pth(self, path):
        d={
            "cp_deform": self.cp_deform,
            "cp_radius": self.cp_radius.detach().clone().cpu(),
            "cp_center": self.cp_center.detach().clone().cpu(),
            "cp_rotation": self.cp_rotation.detach().clone().cpu(),
            "total_time": self.total_time,
            "n_cp_num": self.n_cp_num,
            "c_cp_deform": self.c_cp_deform,
            "c_cp_radius": self.c_cp_radius.detach().clone().cpu() if self.c_cp_radius is not None else None,
            "c_cp_center": self.c_cp_center.detach().clone().cpu() if self.c_cp_center is not None else None,
            "c_cp_rotation": self.c_cp_rotation.detach().clone().cpu() if self.c_cp_rotation is not None else None,
            "mult_rot_way": self.mult_rot_way,
        }
        torch.save(d, path)
        
    def load_pth(self, path):
        d = torch.load(path, weights_only=False)
        self.cp_deform = d["cp_deform"]
        self.cp_center = d["cp_center"].clone().cuda().detach()
        self.cp_radius = nn.Parameter(d["cp_radius"].clone().cuda().requires_grad_(True))
        self.cp_rotation = nn.Parameter(d["cp_rotation"].clone().cuda().requires_grad_(True))
        if "n_cp_num" in d:
            self.n_cp_num = d["n_cp_num"]
        try:
            self.total_time = d["total_time"]
        except:
            self.total_time = self.cp_deform.total_time
        if "c_cp_deform" in d and d["c_cp_deform"] is not None:
            self.c_cp_deform = d["c_cp_deform"]
        if "c_cp_radius" in d and d["c_cp_radius"] is not None:
            self.c_cp_radius = nn.Parameter(d["c_cp_radius"].clone().cuda().requires_grad_(True))
        if "c_cp_center" in d and d["c_cp_center"] is not None:
            self.c_cp_center = d["c_cp_center"].clone().cuda().detach()
        if "c_cp_rotation" in d and d["c_cp_rotation"] is not None:
            self.c_cp_rotation = nn.Parameter(d["c_cp_rotation"].clone().cuda().requires_grad_(True))
        if "mult_rot_way" in d:
            self.mult_rot_way = d["mult_rot_way"]
        self.detach_first_layer = False
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def assign_deform(self, max_time):
        if self.cp_deform.converted_to_independent:
            last_deform = self.cp_deform.cp_deform[max_time].detach().clone()
            new_deform = self.cp_deform.cp_deform.detach().clone()
            for time in range(self.cp_deform.total_time):
                if time <= max_time:
                    continue
                new_deform[time] = last_deform
            optimizable_tensors = self.replace_tensor_to_optimizer(new_deform, "cp_deform")
            self.cp_deform.cp_deform = optimizable_tensors["cp_deform"]
            return
        with torch.no_grad():
            last_deform = self.cp_deform.query_no_activate(max_time)
            new_deform = self.cp_deform.cp_deform.detach().clone()
            for time in range(self.cp_deform.total_time):
                if time <= max_time:
                    continue
                x = time + 1
                cur_deform = torch.zeros((self.cp_deform.cp_num, 7), device="cuda", dtype=torch.float32)
                x = x - lowbit(x)
                while(x > 0):
                    cur_deform = cur_deform + new_deform[x].detach()
                    x = x - lowbit(x)
                cur_deform[:, 0] += 1.0
                delta_deform = last_deform - cur_deform
                new_deform[time + 1] = delta_deform
            optimizable_tensors = self.replace_tensor_to_optimizer(new_deform, "cp_deform")
            self.cp_deform.cp_deform = optimizable_tensors["cp_deform"]
