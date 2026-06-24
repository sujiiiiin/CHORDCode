import os
import math
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from mesh_renderer.mesh import Mesh, safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

class Renderer(nn.Module):
    def __init__(self, opt, resize=True, resize_other_mesh=None):
        
        super().__init__()

        self.opt = opt

        self.mesh = Mesh.load(self.opt.mesh, resize=resize, resize_other_mesh=resize_other_mesh)

        if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
        
        # extract trainable parameters
        if self.mesh.albedo is not None:
            self.raw_albedo = trunc_rev_sigmoid(self.mesh.albedo)
        else:
            self.raw_albedo = None



    @torch.no_grad()
    def export_mesh(self, save_path):
        self.mesh.v = (self.mesh.v).detach()
        self.mesh.albedo = torch.sigmoid(self.raw_albedo.detach())
        self.mesh.write(save_path)

    def sample_points_from_mesh(self, num_points=1000):
        """
        Sample points from the mesh surface (triangles) and get their corresponding colors.
        """
        v = self.mesh.v
        f = self.mesh.f
        num_faces = f.shape[0]

        if num_faces == 0:
            return torch.zeros(num_points, 3, device="cuda"), torch.zeros(num_points, 3, device="cuda")

        # Compute areas of each triangle
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        face_probs = face_areas / face_areas.sum()

        # Sample faces based on area
        face_indices = torch.multinomial(face_probs, num_points, replacement=True)
        chosen_faces = f[face_indices]

        # Generate random barycentric coordinates
        u = torch.sqrt(torch.rand(num_points, 1, device="cuda"))
        v_rand = torch.rand(num_points, 1, device="cuda")
        bary_coords = torch.cat([1 - u, u * (1 - v_rand), u * v_rand], dim=1)

        # Compute sampled points
        sampled_points = (
            bary_coords[:, 0:1] * v[chosen_faces[:, 0]] +
            bary_coords[:, 1:2] * v[chosen_faces[:, 1]] +
            bary_coords[:, 2:3] * v[chosen_faces[:, 2]]
        )

        sampled_colors = None

        # Sample colors
        if self.mesh.vc is not None and self.mesh.vc.shape[0] == v.shape[0]:
            sampled_colors = (
                bary_coords[:, 0:1] * self.mesh.vc[chosen_faces[:, 0]] +
                bary_coords[:, 1:2] * self.mesh.vc[chosen_faces[:, 1]] +
                bary_coords[:, 2:3] * self.mesh.vc[chosen_faces[:, 2]]
            )
        elif self.mesh.albedo is not None and self.mesh.vt is not None and self.mesh.ft is not None:
            vt = self.mesh.vt
            ft = self.mesh.ft
            chosen_ft = ft[face_indices]
            sampled_uvs = (
                bary_coords[:, 0:1] * vt[chosen_ft[:, 0]] +
                bary_coords[:, 1:2] * vt[chosen_ft[:, 1]] +
                bary_coords[:, 2:3] * vt[chosen_ft[:, 2]]
            )
            albedo = torch.sigmoid(self.raw_albedo)
            height, width, _ = albedo.shape
            px = (sampled_uvs[:, 0] * (width - 1)).long().clamp(0, width - 1)
            py = (sampled_uvs[:, 1] * (height - 1)).long().clamp(0, height - 1)
            sampled_colors = albedo[py, px]

        if sampled_colors is None:
            sampled_colors = torch.full_like(sampled_points, 0.5)

        return sampled_points, sampled_colors
    
    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1, texture_filter='linear-mipmap-linear'):
        
        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))

        alpha = (rast[0, ..., 3:] > 0).float()
        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        if self.mesh.vc is not None:
            vc = self.mesh.vc
            # if vertex colors are in 0-255 range, normalize to [0,1]
            if vc.max() > 1.0:
                vc = vc / 255.0
            # debug: print statistics once so we can see if vertex colors are flat
            if not hasattr(self, '_vc_debug_printed'):
                print(f"[DEBUG] vertex color stats min={vc.min().item():.6f}, max={vc.max().item():.6f}, mean={vc.mean().item():.6f}")
                self._vc_debug_printed = True
            albedo, _ = dr.interpolate(vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
            albedo = albedo.clamp(0, 1)
        elif self.raw_albedo is not None:
            texc, texc_db = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft, rast_db=rast_db, diff_attrs='all')
            albedo = dr.texture(self.raw_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]
            albedo = torch.sigmoid(albedo)
        else:
            albedo = torch.full((1, h, w, 3), 0.5, device=v.device)
            # print("using constant albedo")
        # get vn and render normal
        vn = self.mesh.vn
        
        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        # antialias
        albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).squeeze(0) # [H, W, 3]
        albedo = alpha * albedo + (1 - alpha) * bg_color

        # ssaa
        if ssaa != 1:
            albedo = scale_img_hwc(albedo, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = albedo.clamp(0, 1)
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['viewcos'] = viewcos

        return results

    def render_depth3d(self, pose, proj, h0, w0, ssaa=1, bg_color=0.0):

        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0

        results = {}

        # get v
        v = self.mesh.v

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render xyz map
        v_homo = F.pad(v, pad=(0, 1), mode='constant', value=1.0)
        v_cam = torch.matmul(v_homo, torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, _ = dr.rasterize(self.glctx, v_clip, self.mesh.f, (h, w))
        alpha = (rast[0, ..., 3:] > 0).float()

        # Interpolate world/object-space xyz for each pixel.
        xyz, _ = dr.interpolate(v.unsqueeze(0).contiguous(), rast, self.mesh.f)
        xyz = dr.antialias(xyz, rast, v_clip, self.mesh.f).squeeze(0)

        if not torch.is_tensor(bg_color):
            bg_color = torch.tensor(bg_color, dtype=xyz.dtype, device=xyz.device)
        else:
            bg_color = bg_color.to(device=xyz.device, dtype=xyz.dtype)
        if bg_color.numel() == 1:
            bg_color = bg_color.expand(3)
        bg_color = bg_color.view(1, 1, 3)

        xyz = alpha * xyz + (1 - alpha) * bg_color

        # ssaa
        if ssaa != 1:
            xyz = scale_img_hwc(xyz, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))

        results['image'] = xyz
        results['xyz'] = xyz
        results['alpha'] = alpha

        return results
