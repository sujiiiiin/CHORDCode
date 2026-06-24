import copy
import os

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import OptimizationParams
from gaussian_renderer import render_with_cam
from scene import GaussianModel, Scene
from utils.loss_utils import l1_loss, ssim
from utils.orbit_cam_utils import MiniCam, OrbitCamera, orbit_camera


def train_gaussian_by_parts(
    mesh_opt, scene_mesh, mesh_dir_path, opt: OptimizationParams, dataset, saving_iterations
):
    gaussians_list = []
    for root, _, files in os.walk(mesh_dir_path):
        for file in files:
            if not file.endswith(".glb") or file.startswith("scene"):
                continue

            mesh_path = os.path.join(root, file)
            mesh_opt.mesh = mesh_path
            name = file.split(".")[0]

            if opt.use_light:
                print("using light mesh renderer")
                from mesh_renderer.mesh_renderer_light import Renderer
            else:
                from mesh_renderer.mesh_renderer import Renderer

            mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(
                torch.device("cuda")
            )
            save_img_inter = 10 if opt.test else 2000
            points, rgb = mesh_renderer.sample_points_from_mesh(num_points=5_000)

            load_iter = opt.iterations if opt.load_from_checkpoint else None
            if opt.load_from_checkpoint:
                print("split part load iteration: ", load_iter)

            cur_dataset = copy.deepcopy(dataset)
            cur_dataset.model_path = os.path.join(dataset.model_path, name)
            gaussians = GaussianModel(
                dataset.sh_degree, dataset.is_marbles
            )
            scene = Scene(
                cur_dataset,
                gaussians,
                load_iteration=load_iter,
                points=points.cpu().numpy(),
                rgb=rgb.cpu().numpy(),
            )
            gaussians.training_setup(opt)

            if opt.load_from_checkpoint:
                gaussians.update_learning_rate(opt.iterations)
                gaussians_list.append(gaussians)
                continue

            ema_loss_for_log = 0.0
            first_iter = 1
            progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
            default_cam = OrbitCamera(
                dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy
            )

            for iteration in range(first_iter, opt.iterations + 1):
                gaussians.update_learning_rate(iteration)
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                bg_color = 1 if np.random.rand() > opt.invert_bg_prob else 0
                bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda")

                elev = np.random.uniform(opt.elev_l, opt.elev_r)
                azim = np.random.uniform(opt.azim_l, opt.azim_r)
                pose = orbit_camera(-elev, azim, opt.cam_radius, target=None)
                with torch.no_grad():
                    out_i = mesh_renderer.render(
                        pose,
                        default_cam.perspective,
                        dataset.image_height,
                        dataset.image_width,
                        ssaa=1,
                        bg_color=bg_color,
                    )
                    gt_image = (
                        out_i["image"].permute(2, 0, 1).contiguous().unsqueeze(0)
                    )

                cur_gs_cam = MiniCam(
                    pose,
                    dataset.image_width,
                    dataset.image_height,
                    default_cam.fovy,
                    default_cam.fovx,
                    default_cam.near,
                    default_cam.far,
                )
                render_pkg = render_with_cam(cur_gs_cam, gaussians, None, bg)
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]

                Ll1 = l1_loss(image, gt_image[:, :3])
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                    1.0 - ssim(image, gt_image[:, :3])
                )
                loss.backward()

                with torch.no_grad():
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix(
                            {"Loss": f"{ema_loss_for_log:.7f}", "Num": f"{gaussians.get_xyz.shape[0]}"}
                        )
                        progress_bar.update(10)
                    if iteration == opt.iterations:
                        progress_bar.close()

                    if iteration in saving_iterations:
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration, save_ply=True)

                    if iteration < opt.densify_until_iter:
                        gaussians.max_radii2D[visibility_filter] = torch.max(
                            gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter],
                        )
                        gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                            image.shape[2],
                            image.shape[1],
                        )

                        if (
                            iteration > opt.densify_from_iter
                            and iteration % opt.densification_interval == 0
                        ):
                            size_threshold = (
                                20 if iteration > opt.opacity_reset_interval else None
                            )
                            gaussians.densify_and_prune(
                                opt.densify_grad_threshold,
                                0.05,
                                scene.cameras_extent,
                                size_threshold,
                            )

                        if (
                            iteration % opt.opacity_reduce_interval == 0
                            and iteration < opt.opacity_reduce_until_iter
                        ):
                            gaussians.prune(0.05)
                            gaussians.reduce_opacity()

                    if iteration < opt.iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

                    if iteration % save_img_inter == 1:
                        save_path1 = os.path.join(scene.model_path, "train_picures")
                        save_path2 = os.path.join(scene.model_path, "train_picures_gt")
                        os.makedirs(save_path1, exist_ok=True)
                        os.makedirs(save_path2, exist_ok=True)
                        torchvision.utils.save_image(
                            image[0],
                            os.path.join(save_path1, "train_iter_{}.png".format(iteration)),
                        )
                        torchvision.utils.save_image(
                            gt_image[0, :3],
                            os.path.join(save_path2, "train_iter_{}.png".format(iteration)),
                        )

            gaussians_list.append(gaussians)

    part_mask = torch.zeros((gaussians_list[0]._xyz.shape[0]), dtype=torch.int64, device="cuda")
    final_gaussians = gaussians_list[0]
    for i in range(1, len(gaussians_list)):
        cur_gaussians = gaussians_list[i]
        part_mask = torch.cat(
            (
                part_mask,
                torch.ones((cur_gaussians._xyz.shape[0]), dtype=torch.int64, device="cuda") * i,
            ),
            dim=0,
        )
        final_gaussians.densification_postfix(
            cur_gaussians._xyz.detach().clone(),
            cur_gaussians._features_dc.detach().clone(),
            cur_gaussians._features_rest.detach().clone(),
            cur_gaussians._opacity.detach().clone(),
            cur_gaussians._scaling.detach().clone(),
            cur_gaussians._rotation.detach().clone(),
        )
    return final_gaussians, part_mask
