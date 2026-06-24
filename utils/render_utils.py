import math

import torch

from gaussian_renderer.dynamic_renderer import (
    render_dynamic_range,
    render_dynamic_whole,
)
from utils.orbit_cam_utils import MiniCam, orbit_camera


def build_orbit_view(dataset, default_cam, cam_radius, elev, azim, look_at=None):
    pose = orbit_camera(-elev, azim, cam_radius, target=look_at)
    viewpoint_camera = MiniCam(
        pose,
        dataset.image_width,
        dataset.image_height,
        default_cam.fovy,
        default_cam.fovx,
        default_cam.near,
        default_cam.far,
    )
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    opencv_k = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    ).unsqueeze(0)
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)
    return viewmat, opencv_k


def build_view(dataset, default_cam, cam_radius, elev, azim, look_at=None):
    return build_orbit_view(dataset, default_cam, cam_radius, elev, azim, look_at=look_at)


def render_sds_training_video(viewmat, opencv_k, bg_color, dataset, opt, dynamic_scene):
    viewmat = viewmat.unsqueeze(0).repeat(opt.frame_num, 1, 1, 1)
    opencv_k = opencv_k.unsqueeze(0).repeat(opt.frame_num, 1, 1, 1)
    bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda")

    rendered = render_dynamic_whole(
        viewmat,
        opencv_k,
        dynamic_scene,
        dataset.image_width,
        dataset.image_height,
        None,
        opt.frame_num,
        bg,
        detach_radius=opt.detach_radius,
    )
    return rendered[:, 0]


def render_sds_training_video_chunk(
    viewmat,
    opencv_k,
    bg_color,
    dataset,
    opt,
    dynamic_scene,
    start_frame,
    end_frame,
):
    chunk_time = end_frame - start_frame
    viewmat = viewmat.unsqueeze(0).repeat(chunk_time, 1, 1, 1)
    opencv_k = opencv_k.unsqueeze(0).repeat(chunk_time, 1, 1, 1)
    bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda")
    return render_dynamic_range(
        viewmat,
        opencv_k,
        dynamic_scene,
        dataset.image_width,
        dataset.image_height,
        None,
        start_frame,
        end_frame,
        bg,
        detach_radius=opt.detach_radius,
    )[:, 0]


def backward_sds_render_chunks(
    viewmat,
    opencv_k,
    bg_color,
    dataset,
    opt,
    dynamic_scene,
    video_grad,
):
    chunk_size = int(opt.split_render_chunk_size)
    if chunk_size <= 0:
        rendered_video = render_sds_training_video(
            viewmat,
            opencv_k,
            bg_color,
            dataset,
            opt,
            dynamic_scene,
        )
        rendered_video.backward(video_grad)
        del rendered_video
        return

    for start_frame in range(0, opt.frame_num, chunk_size):
        end_frame = min(start_frame + chunk_size, opt.frame_num)
        rendered_chunk = render_sds_training_video_chunk(
            viewmat,
            opencv_k,
            bg_color,
            dataset,
            opt,
            dynamic_scene,
            start_frame,
            end_frame,
        )
        rendered_chunk.backward(video_grad[start_frame:end_frame])
        del rendered_chunk
        torch.cuda.empty_cache()
