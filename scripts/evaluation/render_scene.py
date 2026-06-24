import copy
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional

import math
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, SDSOptimizationParams
from gaussian_renderer.dynamic_renderer import render_dynamic
from scene import GaussianModel, Scene
from scene.dynamic_gaussian_model import DynamicGaussianModel
from scene.dynamic_scene import DynamicGaussianScene
from utils.general_utils import safe_state
from utils.orbit_cam_utils import MiniCam, OrbitCamera, orbit_camera
from utils.scene_object_utils import resolve_scene_obj_num
from wan.utils.utils import cache_video


def _normalize_obj_names(values: Optional[Iterable[str]]) -> set[str]:
    normalized = set()
    for value in values or []:
        value = str(value).strip()
        if not value:
            continue
        normalized.add(value if value.startswith("obj_") else f"obj_{value}")
    return normalized


def _resolve_checkpoint(model_path: str, ex_name: str, checkpoint: int) -> int:
    if checkpoint >= 0:
        return checkpoint

    deform_root = os.path.join(model_path, ex_name, "deform")
    if not os.path.isdir(deform_root):
        raise FileNotFoundError(f"Deformation directory not found: {deform_root}")

    checkpoint_ids = []
    with os.scandir(deform_root) as entries:
        for entry in entries:
            match = re.fullmatch(r"deform_(\d+)", entry.name)
            if match and entry.is_dir():
                checkpoint_ids.append(int(match.group(1)))

    if not checkpoint_ids:
        raise FileNotFoundError(f"No deformation checkpoints found under {deform_root}")
    return max(checkpoint_ids)


def _checkpoint_path(dataset, opt, checkpoint: int, obj_name: str) -> str:
    return os.path.join(
        dataset.model_path,
        opt.ex_name,
        "deform",
        f"deform_{checkpoint}",
        f"{obj_name}.pth",
    )


def _build_view(dataset, cam_radius: float, elev: float, azim: float, cam_height: float):
    default_cam = OrbitCamera(
        dataset.image_width,
        dataset.image_height,
        r=cam_radius,
        fovy=dataset.fovy,
    )
    look_at = np.array([0.0, cam_height, 0.0], dtype=np.float32)
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


def _load_dynamic_scene(
    dataset,
    opt: SDSOptimizationParams,
    checkpoint: int,
    no_bg: bool,
    exclude_objs: Optional[Iterable[str]],
    additional_static_objs: Optional[Iterable[str]],
):
    dynamic_scene = DynamicGaussianScene(opt.frame_num)
    excluded = _normalize_obj_names(exclude_objs)
    forced_static = _normalize_obj_names(additional_static_objs)

    for obj_idx in range(opt.obj_num):
        obj_name = f"obj_{obj_idx}"
        if obj_name in excluded:
            print(f"[Render] Skipping excluded object {obj_name}")
            continue

        cur_dataset = copy.deepcopy(dataset)
        cur_dataset.model_path = os.path.join(dataset.model_path, obj_name)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(cur_dataset, gaussians, load_iteration=-1)

        is_static = obj_idx == opt.static_id or obj_name in forced_static
        if is_static:
            if no_bg and obj_idx == opt.static_id:
                print(f"[Render] Skipping static background {obj_name}")
                continue
            dynamic_scene.add_gaussians(gaussians, True, obj_name)
            continue

        ckpt_path = _checkpoint_path(dataset, opt, checkpoint, obj_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing deformation checkpoint for {obj_name}: {ckpt_path}")

        dynamic_gaussians = DynamicGaussianModel(
            gaussians,
            opt.frame_num,
            scene.cameras_extent,
            n_cp_num=opt.n_cp_num,
        )
        dynamic_gaussians.load_pth(ckpt_path)
        dynamic_scene.add_gaussians(dynamic_gaussians, False, obj_name)

    if not dynamic_scene.dynamic_gaussians:
        raise ValueError("No objects remain to render after applying exclusions.")
    return dynamic_scene


def _frame_count(dynamic_scene: DynamicGaussianScene, lst_frame: Optional[int]) -> int:
    dynamic_times = [
        gaussians.total_time
        for gaussians, is_static in dynamic_scene.dynamic_gaussians.values()
        if not is_static
    ]
    frame_count = min(dynamic_times) if dynamic_times else dynamic_scene.total_time
    if lst_frame is not None:
        if lst_frame <= 0:
            raise ValueError("--lst_frame must be positive when provided")
        frame_count = min(frame_count, lst_frame)
    if frame_count <= 0:
        raise ValueError("Resolved frame count is zero.")
    return frame_count


def _format_view_value(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def render_scene(
    dataset,
    opt: SDSOptimizationParams,
    checkpoint: int,
    azim: float,
    elev: float,
    output_dir: Optional[str],
    lst_frame: Optional[int],
    no_bg: bool,
    exclude_objs: Optional[Iterable[str]],
    additional_static_objs: Optional[Iterable[str]],
):
    checkpoint = _resolve_checkpoint(dataset.model_path, opt.ex_name, checkpoint)
    dynamic_scene = _load_dynamic_scene(
        dataset,
        opt,
        checkpoint,
        no_bg=no_bg,
        exclude_objs=exclude_objs,
        additional_static_objs=additional_static_objs,
    )
    frame_count = _frame_count(dynamic_scene, lst_frame)

    if output_dir is None:
        output_dir = os.path.join(
            dataset.model_path,
            opt.ex_name,
            "rendered_videos",
            f"iteration_{checkpoint}",
        )
    os.makedirs(output_dir, exist_ok=True)

    viewmat, opencv_k = _build_view(dataset, opt.cam_radius, elev, azim, opt.cam_height)
    bg_value = 1.0 if opt.invert_bg_prob <= 0.0 else 0.0
    bg = torch.tensor([bg_value, bg_value, bg_value], dtype=torch.float32, device="cuda")

    frames = []

    with torch.no_grad():
        for time in range(frame_count):
            rgb = render_dynamic(
                viewmat,
                opencv_k,
                dynamic_scene,
                dataset.image_width,
                dataset.image_height,
                None,
                bg,
                time=time,
                detach_radius=opt.detach_radius,
            )
            frames.append(rgb.detach().cpu())

    video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    azim_name = _format_view_value(azim)
    elev_name = _format_view_value(elev)
    video_path = os.path.join(output_dir, f"video_azim_{azim_name}_elev_{elev_name}.mp4")
    saved_video = cache_video(
        tensor=video_tensor,
        save_file=video_path,
        fps=max(float(opt.save_fps), 1.0),
        nrow=1,
        normalize=True,
        value_range=(0, 1),
    )
    if saved_video is None:
        raise RuntimeError(f"Failed to save video to {video_path}")
    print(f"[Render] Saved video to {video_path}")


def main():
    parser = ArgumentParser(description="Render a saved CHORD dynamic checkpoint from one viewpoint")
    lp = ModelParams(parser)
    op = SDSOptimizationParams(parser)
    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint iteration to load; -1 uses the latest")
    parser.add_argument("--azim", type=float, default=None, help="View azimuth in degrees")
    parser.add_argument("--elev", type=float, default=None, help="View elevation in degrees; positive is above the scene")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for the rendered video")
    parser.add_argument("--lst_frame", type=int, default=None, help="Limit the number of rendered frames")
    parser.add_argument("--no_bg", action="store_true", help="Do not render the static background object")
    parser.add_argument("--exclude_objs", nargs="+", type=str, default=[], help="Objects to exclude, e.g. obj_1 or 1")
    parser.add_argument("--additional_static_objs", nargs="+", type=str, default=[], help="Extra objects to treat as static")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    dataset = lp.extract(args)
    opt = op.extract(args)
    resolve_scene_obj_num(opt, dataset.mesh_source_path)
    azim = opt.azim_l if args.azim is None else args.azim
    elev = opt.elev_l if args.elev is None else args.elev

    render_scene(
        dataset,
        opt,
        args.checkpoint,
        azim,
        elev,
        args.output_dir,
        args.lst_frame,
        args.no_bg,
        args.exclude_objs,
        args.additional_static_objs,
    )


if __name__ == "__main__":
    main()
