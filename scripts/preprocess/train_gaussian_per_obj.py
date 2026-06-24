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
import copy
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_with_cam, render_with_cam_depth
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams
import torchvision
import numpy as np
from omegaconf import OmegaConf
from utils.orbit_cam_utils import orbit_camera, OrbitCamera, MiniCam, get_cam_pos_from_pose
import warnings
from utils.train_gs_utils import train_gaussian_by_parts

from utils.phys_loss_kaolin_utils import inside_barriers
from scene.voxel_grid import get_mesh_barrier_in_out

warnings.filterwarnings('ignore')

def training(dataset, opt: OptimizationParams, saving_iterations, checkpoint, zoom_surface_objs):
    first_iter = 0

    mesh_opt = OmegaConf.load(dataset.mesh_config_path)

    meshes = {}
    # walk through the mesh directory and load all meshes
    gaussians_list = []
    scene_list = []
    mesh_renderers = {}
    mem_id = -1
    static_ind = -1
    for root, dirs, files in os.walk(dataset.mesh_source_path):
        for file in files:
            if file.endswith(".glb") and not file.startswith("scene"):
                mesh_path = os.path.join(root, file)
                name = file.split('.')[0]
                part_folder_path = os.path.join(root, name)
                # if name == "scene" or name == "full_scene":
                #     continue
                # if "obj_" not in name or "static_obj" not in name:
                #     continue
                if name != opt.cur_train:
                    continue
                # if name == "obj_1":
                #     continue
                # if name == "obj_2":
                #     continue
                print("Loading mesh from: ", mesh_path)
                if opt.self_norm is False:
                    mesh_opt.mesh = os.path.join(dataset.mesh_source_path, "scene.glb")
                    # auto find mesh from stage 1
                    if opt.use_light:
                        from mesh_renderer.mesh_renderer_light import Renderer
                        scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))
                    else:
                        from mesh_renderer.mesh_renderer import Renderer
                        scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))
                else:
                    if "static_obj" in name:
                        mesh_opt.mesh = os.path.join(dataset.mesh_source_path, "obj_0.glb")
                        # auto find mesh from stage 1
                        if opt.use_light:
                            from mesh_renderer.mesh_renderer_light import Renderer
                            scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))
                        else:
                            from mesh_renderer.mesh_renderer import Renderer
                            scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))
                    else:
                        mesh_opt.mesh = mesh_path
                        # auto find mesh from stage 1
                        if opt.use_light:
                            from mesh_renderer.mesh_renderer_light import Renderer
                            scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))
                        else:
                            from mesh_renderer.mesh_renderer import Renderer
                            scene_mesh = Renderer(mesh_opt, resize=False).to(torch.device("cuda"))

                mesh_opt.mesh = mesh_path
                if name == f"obj_{opt.static_id}":
                    is_static = True
                    static_ind = len(gaussians_list)
                else:
                    is_static = False
                if is_static is False:
                    mem_id = len(gaussians_list)

                if opt.use_light and (not is_static or opt.light_bg):
                    print("using light mesh renderer")
                    from mesh_renderer.mesh_renderer_light import Renderer
                    mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(torch.device("cuda"))
                else:
                    from mesh_renderer.mesh_renderer import Renderer
                    mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(torch.device("cuda"))
                mesh_renderers[name] = mesh_renderer
                rescale_scale = 1.0/mesh_renderer.mesh.get_scale_orig()
                if name in zoom_surface_objs and is_static:
                    from mesh_renderer.mesh_renderer import Renderer
                    mesh_opt.mesh = os.path.join(dataset.mesh_source_path, "static_obj.glb")
                    static_mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(torch.device("cuda"))

                if name in zoom_surface_objs:
                    bars = {}
                    if is_static:
                        in_bar, out_bar = get_mesh_barrier_in_out(static_mesh_renderer.mesh, opt.penetrate_voxel_size/rescale_scale, padding=opt.penetrate_voxel_size/rescale_scale*5.0)
                    else:
                        in_bar, out_bar = get_mesh_barrier_in_out(mesh_renderer.mesh, opt.penetrate_voxel_size/rescale_scale, padding=opt.penetrate_voxel_size/rescale_scale*5.0)
                    bars[name] = (in_bar, out_bar)
                save_img_inter = 2000
                if opt.test:
                    save_img_inter = 10
                points, rgb = mesh_renderer.sample_points_from_mesh(num_points=5_000)
                load_iter = None
                if opt.load_from_checkpoint:
                    load_iter = opt.iterations
                cur_dataset = copy.deepcopy(dataset)
                cur_dataset.model_path = os.path.join(dataset.model_path, name)
                gaussians = GaussianModel(dataset.sh_degree, (dataset.is_marbles and (not is_static)))

                scene = Scene(cur_dataset, gaussians, load_iteration=load_iter
                              , points = points.cpu().numpy(), rgb = rgb.cpu().numpy())
                gaussians.training_setup(opt)
                if opt.load_from_checkpoint:
                    gaussians.update_learning_rate(opt.iterations)
                    gaussians_list.append(gaussians)
                    scene_list.append(scene)
                    continue
                if checkpoint:
                    (model_params, first_iter) = torch.load(checkpoint)
                    gaussians.restore(model_params, opt)
                if os.path.exists(part_folder_path) and opt.train_by_parts:
                    gaussians, part_mask = train_gaussian_by_parts(mesh_opt, scene_mesh, part_folder_path, opt, cur_dataset, saving_iterations)
                    scene = Scene(cur_dataset, gaussians, load_iteration=load_iter
                                  , points = points.cpu().numpy(), rgb = rgb.cpu().numpy(), init=False)
                    scene.save(opt.iterations, save_ply = True)
                    part_mask_path = os.path.join(scene.model_path, "part_mask.npy")
                    np.save(part_mask_path, part_mask.cpu().numpy())
                    gaussians.update_learning_rate(opt.iterations)
                    gaussians_list.append(gaussians)
                    scene_list.append(scene)
                    continue

                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                # iter_start = torch.cuda.Event(enable_timing = True)
                # iter_end = torch.cuda.Event(enable_timing = True)

                viewpoint_stack = None
                feature_stack = None
                ema_loss_for_log = 0.0
                progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
                first_iter += 1
                default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy)
                for iteration in range(first_iter, opt.iterations + 1):

                    gaussians.update_learning_rate(iteration)

                    # Every 1000 its we increase the levels of SH up to a maximum degree
                    if iteration % 1000 == 0:
                        gaussians.oneupSHdegree()

                    if np.random.rand() > opt.invert_bg_prob:
                        bg_color = 1
                    else:
                        bg_color = 0
                    bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda")
                    if name in zoom_surface_objs and iteration > opt.zoom_in_iter and np.random.rand() < opt.cam_fixed_on_obj:

                        elev = np.random.uniform(-89.0, 89.0)
                        azim = np.random.uniform(0.0, 360.0)
                        if is_static is False:
                            scale = mesh_renderer.mesh.get_scale_orig()
                        else:
                            scale = static_mesh_renderer.mesh.get_scale_orig()
                        cam_radius = opt.near_cam_radius * scale

                        default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=cam_radius, fovy=dataset.fovy, near=0.001)
                        # random sample from in_bar
                        if is_static is False:
                            target, _ = mesh_renderer.sample_points_from_mesh(num_points=1)
                        else:
                            target, _ = static_mesh_renderer.sample_points_from_mesh(num_points=1)
                        target = target[0].cpu().numpy()
                        while inside_barriers(get_cam_pos_from_pose(-elev, azim, cam_radius, target=target), bars):
                            elev = np.random.uniform(-89.0, 89.0)
                            azim = np.random.uniform(0.0, 360.0)
                            if is_static is False:
                                target, _ = mesh_renderer.sample_points_from_mesh(num_points=1)
                            else:
                                target, _ = static_mesh_renderer.sample_points_from_mesh(num_points=1)
                            target = target[0].cpu().numpy()
                        pose = orbit_camera(-elev, azim, cam_radius, target=target)
                    else:
                        elev = np.random.uniform(opt.elev_l, opt.elev_r)
                        azim = np.random.uniform(opt.azim_l, opt.azim_r)
                        target = None
                        if opt.move_cam_radius > 0 and is_static:
                            target = np.random.uniform(-opt.move_cam_radius, opt.move_cam_radius, size=(3,))
                            target[1] = 0.0
                        if opt.zoom_in_train is False or is_static:
                            pose = orbit_camera(-elev, azim, opt.cam_radius, target=target)
                        if opt.zoom_in_train is True:
                            target = mesh_renderer.mesh.get_center_orig().cpu().numpy()
                            scale = mesh_renderer.mesh.get_scale_orig()
                            pose = orbit_camera(-elev, azim, opt.cam_radius*scale, target=target)

                    with torch.no_grad():
                        out_i = mesh_renderer.render(pose, default_cam.perspective, dataset.image_height, dataset.image_width, ssaa=1, bg_color=bg_color)
                        gt_image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    cur_gs_cam = MiniCam(pose, dataset.image_width, dataset.image_height, default_cam.fovy, default_cam.fovx, default_cam.near, default_cam.far)
                    render_pkg = render_with_cam(cur_gs_cam, gaussians, None, bg)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                    Ll1 = l1_loss(image, gt_image[:,:3])
                    img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image[:,:3]))


                    loss = img_loss

                    if opt.lambda_depth > 0.0 and (is_static or opt.train_with_depth):
                        rendered_depth, _ = render_with_cam_depth(cur_gs_cam, gaussians, None, bg)
                        gt_depth = out_i["depth"].permute(2,0,1).contiguous().unsqueeze(0)  # [1, H, W]
                        loss = loss + opt.lambda_depth * l1_loss(rendered_depth, gt_depth)

                    if opt.lambda_scale > 0.0:
                        scaling_reg = gaussians.get_scaling.prod(dim=1).mean()
                        loss = loss + opt.lambda_scale * scaling_reg

                    loss.backward()

                    #iter_end.record()

                    with torch.no_grad():
                        # Progress bar
                        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                        if iteration % 10 == 0:
                            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Num": f"{gaussians.get_xyz.shape[0]}"})
                            progress_bar.update(10)
                        if iteration == opt.iterations:
                            progress_bar.close()

                        # Log and save
                        if (iteration in saving_iterations):
                            print("\n[ITER {}] Saving Gaussians".format(iteration))
                            # mem = torch.cuda.max_memory_allocated() / 1024**3
                            scene.save(iteration, save_ply = True)

                        # Densification
                        if iteration < opt.densify_until_iter:
                            # Keep track of max radii in image-space for pruning
                            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                                if opt.max_scale > 0.0:
                                    gaussians.densify_by_scale(opt.max_scale * mesh_renderer.mesh.get_scale_orig())


                            if iteration % opt.opacity_reduce_interval == 0 and iteration < opt.opacity_reduce_until_iter:
                                gaussians.prune(0.05)
                                gaussians.reduce_opacity()

                        # Optimizer step
                        if iteration < opt.iterations:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)

                        if iteration % save_img_inter == 1:
                            save_path1 = os.path.join(scene.model_path, "train_picures")
                            save_path2 = os.path.join(scene.model_path, "train_picures_gt")
                            os.makedirs(save_path1, exist_ok=True)
                            os.makedirs(save_path2, exist_ok=True)
                            torchvision.utils.save_image(image[0], os.path.join(save_path1, "train_iter_{}.png".format(iteration)))
                            torchvision.utils.save_image(gt_image[0,:3], os.path.join(save_path2, "train_iter_{}.png".format(iteration)))
                gaussians_list.append(gaussians)
                scene_list.append(scene)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Per-object Gaussian training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--zoom_surface_objs", nargs="+", type=str, default=[])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Training per-object Gaussians for {args.cur_train} in {args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), args.save_iterations, args.start_checkpoint, args.zoom_surface_objs)

    # All done
    print("\nPer-object Gaussian training complete.")
