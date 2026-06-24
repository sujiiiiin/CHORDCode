#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import math

import torch
from gsplat import rasterization

from scene.gaussian_model import GaussianModel


def _camera_intrinsics(viewpoint_camera):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    return torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )


def _expand_background(bg_color: torch.Tensor, target_shape):
    while bg_color.dim() < len(target_shape) + 1:
        bg_color = bg_color.unsqueeze(0)
    return bg_color.expand(*target_shape, bg_color.shape[-1])


def _move_channels_first(tensor: torch.Tensor):
    return torch.movedim(tensor, -1, -3)


def _reduce_radii(radii: torch.Tensor):
    radii = radii.squeeze()
    if radii.dim() == 0:
        return radii.unsqueeze(0)
    if radii.dim() > 1 and radii.shape[-1] == 2:
        radii = radii.max(dim=-1).values
    if radii.dim() > 1:
        radii = radii.max(dim=0).values
    return radii


def _package_render(render_colors, render_alphas, info, retain_grad):
    radii = _reduce_radii(info["radii"])
    if retain_grad:
        try:
            info["means2d"].retain_grad()
        except Exception:
            pass
    return {
        "render": _move_channels_first(render_colors),
        "render_alphas": _move_channels_first(render_alphas),
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "info": info,
    }


def render_with_cam(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    k_matrix = _camera_intrinsics(viewpoint_camera)
    means3d = pc.get_xyz.cuda()
    opacity = pc.get_opacity.cuda()
    scales = pc.get_scaling.cuda() * scaling_modifier
    rotations = pc.get_rotation.cuda()
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features.cuda()
        sh_degree = pc.active_sh_degree

    render_colors, render_alphas, info = rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewpoint_camera.world_view_transform.transpose(0, 1)[None],
        Ks=k_matrix[None],
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        absgrad=True,
        rasterize_mode="antialiased",
    )
    return _package_render(render_colors, render_alphas, info, retain_grad=True)


def render_with_cam_depth(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    k_matrix = _camera_intrinsics(viewpoint_camera)
    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features
        sh_degree = pc.active_sh_degree

    render_colors, render_alphas, _ = rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=viewpoint_camera.world_view_transform.transpose(0, 1)[None],
        Ks=k_matrix[None],
        backgrounds=bg_color[:1][None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        absgrad=True,
        rasterize_mode="antialiased",
        render_mode="ED",
    )
    return _move_channels_first(render_colors), _move_channels_first(render_alphas)


def render_batch(
    view_mats,
    Ks,
    pc: GaussianModel,
    image_width,
    image_height,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    detach=False,
):
    if view_mats.dim() == 2:
        view_mats = view_mats.unsqueeze(0)
    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)

    means3d = pc.get_xyz.cuda()
    rotations = pc.get_rotation.cuda()
    opacity = pc.get_opacity.cuda()
    scales = pc.get_scaling.cuda() * scaling_modifier
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features.cuda()
        sh_degree = pc.active_sh_degree

    if detach:
        means3d = means3d.detach()
        rotations = rotations.detach()
        opacity = opacity.detach()
        scales = scales.detach()
        view_mats = view_mats.detach()
        Ks = Ks.detach()
        bg_color = bg_color.detach()

    render_colors, render_alphas, info = rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=view_mats,
        Ks=Ks,
        backgrounds=_expand_background(bg_color, Ks.shape[:-2]),
        width=image_width,
        height=image_height,
        packed=False,
        sh_degree=sh_degree,
        absgrad=True,
        rasterize_mode="antialiased",
    )
    return _package_render(render_colors, render_alphas, info, retain_grad=not detach)
