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

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from gaussian_renderer.dynamic_renderer import render_dynamic_with_alpha
import warnings

warnings.filterwarnings('ignore')
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, SDSOptimizationParams
import torchvision

from utils.orbit_cam_utils import orbit_camera, OrbitCamera, MiniCam
from utils.scene_object_utils import resolve_scene_obj_num
import math
from scene.dynamic_scene import DynamicGaussianScene
import copy

def training(dataset, opt: SDSOptimizationParams, checkpoint, no_bg, save_rgba):
    dynamic_scene = DynamicGaussianScene(opt.frame_num)
    for i in range(opt.obj_num):
        print("Adding object {}".format(i))
        cur_dataset = copy.deepcopy(dataset)
        cur_dataset.model_path = os.path.join(dataset.model_path, f"obj_{i}")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(cur_dataset, gaussians, load_iteration=-1)
        if i == opt.static_id:
            if no_bg:
                continue
            dynamic_scene.add_gaussians(gaussians, True, "obj_{}".format(i))
            continue
        from scene.dynamic_gaussian_model import DynamicGaussianModel

        dynamic_gaussians = DynamicGaussianModel(
            gaussians,
            opt.frame_num,
            scene.cameras_extent,
            n_cp_num=opt.n_cp_num,
        )
        dynamic_gaussians.init_control_points_from_gaussians(30)
        dynamic_scene.add_gaussians(dynamic_gaussians, False, "obj_{}".format(i))

    dataset.model_path = os.path.join(dataset.model_path)
    os.makedirs(dataset.model_path, exist_ok=True)
    video_path = os.path.join(dataset.model_path, "rendered_images", f"iteration_{checkpoint}")
    #os.makedirs(frame_path, exist_ok=True)

    # print("Print gs number: ", gaussians.get_xyz.shape[0])
    azims = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 240, 270, 300, 300, 330, 345]
    if opt.invert_bg_prob <=0.0:
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    else:
        bg = 1.0 - torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy)
    with torch.no_grad():
        for azim in azims:
            elev = opt.elev_l
            pose = orbit_camera(-elev, azim, opt.cam_radius)
            viewpoint_camera = MiniCam(pose, dataset.image_width, dataset.image_height, default_cam.fovy, default_cam.fovx, default_cam.near, default_cam.far)
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
            opencv_K = torch.tensor(
                [
                    [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                    [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                    [0, 0, 1],
                ],
                device="cuda",
            ).unsqueeze(0)
            viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0)

            fps = 10
            frames = []

            os.makedirs(video_path, exist_ok=True)
            for i in range(dynamic_scene.total_time):
                # print("print frame time: ", i)
                cur_frame, cur_alpha = render_dynamic_with_alpha(viewmat, opencv_K, dynamic_scene
                                                    , dataset.image_width, dataset.image_height, None, bg, time=i)
                if save_rgba:
                    cur_frame = torch.cat((cur_frame, cur_alpha), dim=0)
                #torchvision.utils.save_image(cur_frame, os.path.join(frame_path, f"frame_{azim}_{i}.png"))
                frames.append(cur_frame)
            first_frame = frames[0]
            torchvision.utils.save_image(first_frame, os.path.join(video_path, f"frame_{azim}_0.png"))
            frames = torch.stack(frames, dim=0)
            frames = frames.permute(1, 0, 2, 3)
            # print("print video path: ", os.path.join(video_path, f"video_{azim}.mp4"))
            # cache_video(
            #                 tensor=frames[None],
            #                 save_file=os.path.join(video_path, f"video_{azim}.mp4"),
            #                 fps=fps,
            #                 nrow=1,
            #                 normalize=True,
            #                 value_range=(0, 1))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = SDSOptimizationParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no_bg", action="store_true", help="Do not use background in the model")
    parser.add_argument("--save_rgba", action="store_true", help="Save rgba images")
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.cuda.set_device(args.device)
    dataset = lp.extract(args)
    opt = op.extract(args)
    resolve_scene_obj_num(opt, dataset.mesh_source_path)
    training(dataset, opt, args.checkpoint, args.no_bg, args.save_rgba)

    # All done
    print("\nTraining complete.")
