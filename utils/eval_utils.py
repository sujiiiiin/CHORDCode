import os

import torch

from gaussian_renderer.dynamic_renderer import render_cp_with_mask, render_dynamic
from gaussian_renderer.voxel_render import render_voxel_set
from utils.render_utils import build_view
from wan.utils.utils import cache_video


def save_cp_orbit(dataset, opt, gaussians, video_path, default_cam, additional=False, look_at=None):
    rendered_normal_frames = []
    total_views = 50
    for i in range(total_views):
        elev = (opt.elev_l + opt.elev_r) * 0.5
        azim = i * 360.0 / total_views
        viewmat, opencv_k = build_view(
            dataset,
            default_cam,
            opt.cam_radius,
            elev,
            azim,
            look_at=look_at,
        )
        bg = 1.0 - torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        cp_frame, cp_alpha = render_cp_with_mask(
            viewmat,
            opencv_k,
            gaussians,
            dataset.image_width,
            dataset.image_height,
            None,
            bg,
            time=0,
            scaling_modifier=0.3,
            opacity_factor=0.7,
            additional=additional,
        )
        cur_normal_frame = render_dynamic(
            viewmat,
            opencv_k,
            gaussians,
            dataset.image_width,
            dataset.image_height,
            None,
            1.0 - bg,
            time=0,
        )
        cur_normal_frame = cur_normal_frame * (1.0 - cp_alpha[0]) + cp_frame[0] * cp_alpha[0]
        rendered_normal_frames.append(cur_normal_frame)
    rendered_normal_frames = torch.stack(rendered_normal_frames, dim=0)
    rendered_normal_frames = rendered_normal_frames.permute(1, 0, 2, 3)
    cache_video(
        tensor=rendered_normal_frames[None],
        save_file=video_path,
        fps=20,
        nrow=1,
        normalize=True,
        value_range=(0, 1),
    )


def save_cp_deform(viewmat, opencv_k, dataset, gaussians, video_path, scaling_modifier=1.0):
    video = []
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    cur_time = gaussians.total_time
    for i in range(cur_time):
        normal_frame = render_dynamic(
            viewmat,
            opencv_k,
            gaussians,
            dataset.image_width,
            dataset.image_height,
            None,
            bg,
            time=i,
        )
        video.append(normal_frame)
    video = torch.stack(video, dim=0)
    video = video.permute(1, 0, 2, 3)
    save_fps = 24 if cur_time > 41 else 10
    cache_video(
        tensor=video.squeeze(1)[None],
        save_file=video_path,
        fps=save_fps,
        nrow=1,
        normalize=True,
        value_range=(0, 1),
    )


def render_voxel_grid(dataset, opt, voxel_grid, video_path, default_cam, lid_id=None):
    total_views = 50
    rendered_frames = []
    for i in range(total_views):
        elev = (opt.elev_l + opt.elev_r) * 0.5
        azim = i * 360.0 / total_views
        viewmat, opencv_k = build_view(dataset, default_cam, opt.cam_radius, elev, azim)
        bg = 1.0 - torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
        cur_frame = render_voxel_set(
            viewmat,
            opencv_k,
            voxel_grid,
            dataset.image_width,
            dataset.image_height,
            bg,
            lid_id=lid_id,
        )
        rendered_frames.append(cur_frame)
    rendered_frames = torch.stack(rendered_frames, dim=0)
    rendered_frames = rendered_frames.permute(1, 0, 2, 3)
    cache_video(
        tensor=rendered_frames[None],
        save_file=video_path,
        fps=20,
        nrow=1,
        normalize=True,
        value_range=(0, 1),
    )


def save_iteration_outputs(
    iteration,
    args,
    dataset,
    opt,
    dynamic_scene,
    default_cam,
    ref_viewmat,
    ref_opencv_k,
    cp_deform_path,
    added_acp,
    look_at,
    ddp,
):
    if not ddp.is_main:
        return
    deform_path = os.path.join(dataset.model_path, "deform", f"deform_{iteration}")
    cp_orbit_path = os.path.join(dataset.model_path, "orbit_videos")
    os.makedirs(deform_path, exist_ok=True)
    os.makedirs(cp_orbit_path, exist_ok=True)
    print(f"[CHORD] Saving checkpoint at iteration {iteration}")
    dynamic_scene.save_pth(deform_path)
    save_cp_deform(
        ref_viewmat,
        ref_opencv_k,
        dataset,
        dynamic_scene,
        os.path.join(cp_deform_path, f"deform_video_{iteration}.mp4"),
        scaling_modifier=0.5,
    )
    save_cp_orbit(
        dataset,
        opt,
        dynamic_scene,
        os.path.join(cp_orbit_path, f"orbit_video_{iteration}.mp4"),
        default_cam,
        look_at=look_at,
    )
    if added_acp:
        save_cp_orbit(
            dataset,
            opt,
            dynamic_scene,
            os.path.join(cp_orbit_path, f"acp_orbit_video_{iteration}.mp4"),
            default_cam,
            additional=True,
        )
