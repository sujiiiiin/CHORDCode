import torch
from gsplat import rasterization

from utils.label_utils import num2rgb


def _expand_background(bg_color: torch.Tensor, target_shape):
    while bg_color.dim() < len(target_shape) + 1:
        bg_color = bg_color.unsqueeze(0)
    return bg_color.expand(*target_shape, bg_color.shape[-1])


def _move_channels_first(tensor: torch.Tensor):
    return torch.movedim(tensor, -1, -3)


def render_pc(
    view_mats,
    Ks,
    pc,
    base_voxel_size=0.01,
    image_width=None,
    image_height=None,
    bg_color: torch.Tensor = None,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    lid_id=None,
):
    if view_mats.dim() == 2:
        view_mats = view_mats.unsqueeze(0)
    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)

    means3d = pc if mask is None else pc[mask]
    opacity = torch.ones_like(means3d[:, 0:1], device="cuda")
    scales = torch.ones_like(means3d[:, 0:3], device="cuda") * base_voxel_size * 0.1
    rotations = torch.zeros((means3d.shape[0], 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1.0
    if lid_id is None:
        colors = torch.ones_like(means3d[:, 0:3], device="cuda")
    else:
        colors = num2rgb(lid_id.max().item() + 1)[lid_id]
    if override_color is not None:
        colors = override_color

    render_colors, render_alphas, _ = rasterization(
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
        sh_degree=None,
        absgrad=True,
    )
    render_colors = _move_channels_first(render_colors)
    render_alphas = _move_channels_first(render_alphas)
    return render_colors[0], render_alphas[0]


def render_voxel_set(
    view_mats,
    Ks,
    pcs,
    image_width,
    image_height,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    lid_id=None,
):
    if view_mats.dim() == 2:
        view_mats = view_mats.unsqueeze(0)
    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)

    means3d_parts = []
    opacity_parts = []
    scale_parts = []
    rotation_parts = []
    color_parts = []
    for key, pc in pcs.items():
        if pc is None:
            continue
        try:
            means3d = pc.occupy_xyz
            base_voxel_size = pc.base_voxel_size
        except Exception:
            means3d = pc
            base_voxel_size = 0.015
        if mask is not None:
            means3d = means3d[mask]

        means3d_parts.append(means3d)
        opacity_parts.append(torch.ones_like(means3d[:, 0:1], device="cuda"))
        scale_parts.append(torch.ones_like(means3d[:, 0:3], device="cuda") * base_voxel_size * 0.1)
        rotations = torch.zeros((means3d.shape[0], 4), dtype=torch.float32, device="cuda")
        rotations[:, 0] = 1.0
        rotation_parts.append(rotations)
        if lid_id is None:
            color_parts.append(torch.ones_like(means3d[:, 0:3], device="cuda"))
        else:
            color_parts.append(num2rgb(lid_id[key].max().item() + 1)[lid_id[key]])

    means3d = torch.cat(means3d_parts, dim=0)
    opacity = torch.cat(opacity_parts, dim=0)
    scales = torch.cat(scale_parts, dim=0)
    rotations = torch.cat(rotation_parts, dim=0)
    colors = torch.cat(color_parts, dim=0)
    if override_color is not None:
        colors = override_color

    render_colors, _, _ = rasterization(
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
        sh_degree=None,
        absgrad=True,
    )
    return _move_channels_first(render_colors)[0]
