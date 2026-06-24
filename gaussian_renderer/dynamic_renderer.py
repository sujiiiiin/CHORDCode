import torch
from gsplat import rasterization

from utils.label_utils import num2rgb


def _expand_background(bg_color: torch.Tensor, target_shape):
    while bg_color.dim() < len(target_shape) + 1:
        bg_color = bg_color.unsqueeze(0)
    return bg_color.expand(*target_shape, bg_color.shape[-1])


def _move_channels_first(tensor: torch.Tensor):
    return torch.movedim(tensor, -1, -3)


def _dynamic_sh_degree(pc, default=3):
    try:
        return pc.gaussians.active_sh_degree
    except Exception:
        return default


def _prepare_dynamic_inputs(view_mats, Ks, bg_color):
    if view_mats.dim() == 2:
        view_mats = view_mats.unsqueeze(0)
    if Ks.dim() == 2:
        Ks = Ks.unsqueeze(0)
    return view_mats, Ks, _expand_background(bg_color, Ks.shape[:-2])


def _rasterize(
    means3d,
    rotations,
    scales,
    opacity,
    colors,
    view_mats,
    Ks,
    bg_color,
    image_width,
    image_height,
    sh_degree,
):
    return rasterization(
        means=means3d,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(-1),
        colors=colors,
        viewmats=view_mats,
        Ks=Ks,
        backgrounds=bg_color,
        width=image_width,
        height=image_height,
        packed=False,
        sh_degree=sh_degree,
        absgrad=True,
        rasterize_mode="antialiased",
    )


def render_dynamic_whole(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    max_time,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    detach=False,
    detach_radius=True,
    mem_means=None,
    mem_rotations=None,
):
    if mem_means is not None and mem_rotations is not None:
        means3d = mem_means
        rotations = mem_rotations
    else:
        means3d, rotations = pc.get_xyz_rotation_whole(detach_radius)
    means3d = means3d[:max_time]
    rotations = rotations[:max_time]
    opacity = pc.get_opacity(0).unsqueeze(0).repeat(max_time, 1, 1)
    scales = pc.get_scaling(0).unsqueeze(0).repeat(max_time, 1, 1) * scaling_modifier
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features(0).unsqueeze(0).repeat(max_time, 1, 1, 1)
        sh_degree = _dynamic_sh_degree(pc)

    if detach:
        means3d = means3d.detach().clone()
        rotations = rotations.detach().clone()
        opacity = opacity.detach().clone()
        scales = scales.detach().clone()

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, _, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        sh_degree,
    )
    return _move_channels_first(render_colors)


def render_dynamic_range(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    start_time,
    end_time,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    detach=False,
    detach_radius=True,
):
    _ = pipe
    chunk_time = end_time - start_time
    means3d, rotations = pc.get_xyz_rotation_range(start_time, end_time, detach_radius)
    opacity = pc.get_opacity(0).unsqueeze(0).repeat(chunk_time, 1, 1)
    scales = pc.get_scaling(0).unsqueeze(0).repeat(chunk_time, 1, 1) * scaling_modifier
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features(0).unsqueeze(0).repeat(chunk_time, 1, 1, 1)
        sh_degree = _dynamic_sh_degree(pc)

    if detach:
        means3d = means3d.detach().clone()
        rotations = rotations.detach().clone()
        opacity = opacity.detach().clone()
        scales = scales.detach().clone()

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, _, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        sh_degree,
    )
    return _move_channels_first(render_colors)


def render_dynamic(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    time=None,
    detach=False,
    detach_radius=False,
):
    means3d, rotations = pc.get_xyz_rotation(time, detach_radius)
    opacity = pc.get_opacity(time)
    scales = pc.get_scaling(time) * scaling_modifier
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features(time)
        sh_degree = _dynamic_sh_degree(pc)

    if detach:
        means3d = means3d.detach().clone()
        rotations = rotations.detach().clone()
        opacity = opacity.detach().clone()
        scales = scales.detach().clone()

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, _, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        sh_degree,
    )
    return _move_channels_first(render_colors)[0]


def render_dynamic_with_alpha(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    time=None,
    detach=False,
    detach_radius=False,
):
    means3d, rotations = pc.get_xyz_rotation(time, detach_radius)
    opacity = pc.get_opacity(time)
    scales = pc.get_scaling(time) * scaling_modifier
    if override_color is not None:
        colors = override_color
        sh_degree = None
    else:
        colors = pc.get_features(time)
        sh_degree = _dynamic_sh_degree(pc)

    if detach:
        means3d = means3d.detach().clone()
        rotations = rotations.detach().clone()
        opacity = opacity.detach().clone()
        scales = scales.detach().clone()

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, render_alphas, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        sh_degree,
    )
    render_colors = _move_channels_first(render_colors)
    render_alphas = _move_channels_first(render_alphas)
    return render_colors[0], render_alphas[0]


def render_cp(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    time=None,
    detach=False,
):
    means3d = pc.get_cp_position(time)
    rotations = pc.get_cp_rotation(time)
    opacity = torch.ones((means3d.shape[0], 1), dtype=torch.float32, device="cuda")
    scales = pc.get_cp_scaling() * scaling_modifier
    colors = override_color if override_color is not None else num2rgb(means3d.shape[0])

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, _, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        None,
    )
    return _move_channels_first(render_colors)


def render_cp_with_mask(
    view_mats,
    Ks,
    pc,
    image_width,
    image_height,
    pipe,
    bg_color: torch.Tensor,
    mask=None,
    scaling_modifier=1.0,
    override_color=None,
    time=None,
    detach=False,
    opacity_factor=1.0,
    additional=False,
):
    if additional:
        means3d = pc.get_additional_cp_position(time).float()
        rotations = pc.get_additional_cp_rotation(time).float()
        scales = pc.get_additional_cp_scaling() * scaling_modifier
    else:
        means3d = pc.get_cp_position(time).float()
        rotations = pc.get_cp_rotation(time).float()
        scales = pc.get_cp_scaling() * scaling_modifier

    opacity = torch.ones((means3d.shape[0], 1), dtype=torch.float32, device="cuda") * opacity_factor
    colors = override_color if override_color is not None else num2rgb(means3d.shape[0])

    if mask is not None:
        means3d = means3d[mask]
        rotations = rotations[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        colors = colors[mask]

    view_mats, Ks, bg_color = _prepare_dynamic_inputs(view_mats, Ks, bg_color)
    render_colors, render_alphas, _ = _rasterize(
        means3d,
        rotations,
        scales,
        opacity,
        colors,
        view_mats,
        Ks,
        bg_color,
        image_width,
        image_height,
        None,
    )
    return _move_channels_first(render_colors), _move_channels_first(render_alphas)
