import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import copy
import logging
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

import wan
from arguments import ModelParams, SDSOptimizationParams
from mesh_renderer.mesh_load import Renderer
from scene import GaussianModel, Scene
from scene.dynamic_gaussian_model import DynamicGaussianModel
from scene.dynamic_scene import DynamicGaussianScene
from utils.orbit_cam_utils import OrbitCamera
from utils.ddp_utils import (
    broadcast_scene_parameters,
    extract_dynamic_payload,
    init_distributed_additional_cp,
    install_dynamic_payload,
    sync_gradients,
)
from utils.eval_utils import (
    render_voxel_grid,
    save_cp_deform,
    save_cp_orbit,
    save_iteration_outputs,
)
from utils.optimization_utils import (
    backward_regularization,
    backward_sds,
    build_local_batch,
)
from utils.render_utils import build_view
from utils.resolution_loss_utils import set_resolution_loss_scales
from utils.runtime_utils import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    init_logging,
    mean_metric,
    to_segment_pairs,
)
from utils.scene_object_utils import resolve_scene_obj_num
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."
    elif args.use_tiny_vae:
        raise ValueError("--use_tiny_vae is currently supported only for task i2v-A14B.")

    if args.frame_num is None:
        cfg = WAN_CONFIGS[args.task]
        args.frame_num = cfg.frame_num

    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"



def _validate_release_inputs(args):
    if len(args.azim_segments) % 2 != 0:
        raise ValueError("--azim_segments must be provided as [start end] pairs.")
    if len(args.elev_segments) % 2 != 0:
        raise ValueError("--elev_segments must be provided as [start end] pairs.")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    lp = ModelParams(parser)
    op = SDSOptimizationParams(parser)
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Base random seed for CHORD. A negative value leaves RNGs unseeded; non-negative values use seed + rank.",
    )
    
    parser.add_argument("--gl_objects", nargs="+", type=str, default=[])
    
    parser.add_argument("--cp_nums", nargs="+", type=int, default=[])
    
    parser.add_argument("--time_loss_landmarks", nargs="+", type=float, default=[])
    parser.add_argument("--arap_landmarks", nargs="+", type=float, default=[])
    parser.add_argument("--add_cp_objs", nargs="+", type=str, default=[])

    parser.add_argument("--azim_segments", nargs="+", type=float, default=[])
    parser.add_argument("--elev_segments", nargs="+", type=float, default=[])
    parser.add_argument("--additional_static_objs", nargs="+", type=str, default=[])
    parser.add_argument("--cp_surface_only_objs", nargs="+", type=str, default=[])
    parser.add_argument("--avoid_objs", nargs="+", type=str, default=[])
    args = parser.parse_args()

    _validate_args(args)
    _validate_release_inputs(args)

    return args, lp.extract(args), op.extract(args)

def generate_distributed(args, dataset, opt: SDSOptimizationParams, ddp):
    init_logging(ddp.rank)

    if args.ulysses_size > 1 or args.t5_fsdp or args.dit_fsdp:
        raise ValueError(
            "train.py does not support --ulysses_size > 1, "
            "--t5_fsdp, or --dit_fsdp."
        )

    cfg = WAN_CONFIGS[args.task]
    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]

    if len(args.time_loss_landmarks) != 0:
        opt.lambda_dis_landmarks = args.time_loss_landmarks
    if len(args.arap_landmarks) != 0:
        opt.lambda_arap_landmarks = args.arap_landmarks

    resolve_scene_obj_num(opt, dataset.mesh_source_path)

    if args.cp_nums and len(args.cp_nums) != opt.obj_num:
        raise ValueError("--cp_nums must either be empty or provide one value per object.")

    if len(args.azim_segments) > 0:
        args.azim_segments = to_segment_pairs(args.azim_segments)
        if ddp.is_main:
            print(f"[CHORD] Azimuth sampling segments: {args.azim_segments}")
    if len(args.elev_segments) > 0:
        args.elev_segments = to_segment_pairs(args.elev_segments)
        if ddp.is_main:
            print(f"[CHORD] Elevation sampling segments: {args.elev_segments}")

    mesh_opt = OmegaConf.load(dataset.mesh_config_path)
    mesh_opt.mesh = os.path.join(dataset.mesh_source_path, "scene.glb")
    scene_mesh = Renderer(mesh_opt, resize=False).to(ddp.device)

    dynamic_scene = DynamicGaussianScene(opt.frame_num)
    mem_voxel_xyzs = {}
    saved_voxel_grids = {}
    saved_meshes = {}
    mem_voxel_size = {}
    dynamic_objs = []
    default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy)

    true_ground_level = 10000.0
    mesh_renderer = None
    for i in range(opt.obj_num):
        obj_name = f"obj_{i}"
        if len(args.cp_nums) > 0:
            opt.cp_num = args.cp_nums[i]
            if ddp.is_main:
                print(f"[CHORD] Using {opt.cp_num} control points for {obj_name}")

        cur_dataset = copy.deepcopy(dataset)
        cur_dataset.model_path = os.path.join(dataset.model_path, obj_name)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(cur_dataset, gaussians, load_iteration=-1)
        gaussians.detach_all()
        if ddp.is_main:
            print(f"[CHORD] Preparing {obj_name}")
        if obj_name in args.avoid_objs:
            if ddp.is_main:
                print(f"[CHORD] Skipping avoided object {obj_name}")
            continue

        if i == opt.static_id or obj_name in args.additional_static_objs:
            dynamic_scene.add_gaussians(gaussians, True, obj_name)
            mesh_opt.mesh = os.path.join(dataset.mesh_source_path, f"{obj_name}.glb")
            mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(ddp.device)
            true_ground_level = mesh_renderer.mesh.v[:, 1].min().item()
            continue

        mesh_opt.mesh = os.path.join(dataset.mesh_source_path, f"{obj_name}.glb")
        mesh_renderer = Renderer(mesh_opt, resize_other_mesh=scene_mesh.mesh).to(ddp.device)
        saved_meshes[obj_name] = mesh_renderer.mesh
        dynamic_gaussians = DynamicGaussianModel(
            gaussians,
            opt.frame_num,
            scene.cameras_extent,
            n_cp_num=opt.n_cp_num,
        )
        obj_scale = 1.0 / mesh_renderer.mesh.get_scale_orig()
        mem_voxel_size[obj_name] = opt.init_voxel_size / obj_scale
        cp_surface_only = obj_name in args.cp_surface_only_objs

        payload = None
        if ddp.is_main:
            if cp_surface_only:
                print(f"[CHORD] Using surface-only control-point sampling for {obj_name}")
            print(f"[CHORD] Initializing {obj_name} control points from voxels")
            dynamic_gaussians.init_control_points(
                opt.cp_num,
                init_mesh=mesh_renderer.mesh,
                init_voxel_size=opt.init_voxel_size / obj_scale,
                br_range=opt.br_range,
                cp_surface_only=cp_surface_only,
                target_num=opt.surface_target_num,
            )
            saved_voxel_grids[obj_name] = dynamic_gaussians.voxel_grid
            payload = extract_dynamic_payload(dynamic_gaussians)

        payload = broadcast_object(payload, ddp)
        if not ddp.is_main:
            install_dynamic_payload(dynamic_gaussians, payload)

        mem_voxel_size[obj_name] = payload["mem_init_voxel_size"]
        mem_voxel_xyzs[obj_name] = payload["voxel_xyz"].clone().cuda()
        true_ground_level = min(true_ground_level, mem_voxel_xyzs[obj_name][:, 1].min().item())

        dynamic_gaussians.training_setup(opt)
        dynamic_scene.add_gaussians(dynamic_gaussians, False, obj_name)
        dynamic_objs.append(obj_name)

    true_ground_level_tensor = torch.tensor([true_ground_level], device=ddp.device)
    if ddp.distributed:
        dist.broadcast(true_ground_level_tensor, src=0)
    true_ground_level = true_ground_level_tensor.item()

    ground_levels = {}
    ground_masks = {}
    for key, voxel_xyz in mem_voxel_xyzs.items():
        ground_levels[key] = true_ground_level
        ground_masks[key] = voxel_xyz[:, 1] <= true_ground_level

    dataset.model_path = os.path.join(dataset.model_path, opt.ex_name)
    if ddp.is_main:
        os.makedirs(dataset.model_path, exist_ok=True)
        render_voxel_grid(
            dataset,
            opt,
            saved_voxel_grids,
            os.path.join(dataset.model_path, "voxel_mask_orbit_video.mp4"),
            default_cam,
            lid_id=None,
        )
        if opt.fix_ground:
            ground_id = {}
            for key, ground_mask in ground_masks.items():
                new_ground_id = torch.zeros_like(ground_mask, dtype=torch.int64, device="cuda")
                if key in args.gl_objects:
                    new_ground_id[ground_mask] = 1
                ground_id[key] = new_ground_id
            render_voxel_grid(
                dataset,
                opt,
                saved_voxel_grids,
                os.path.join(dataset.model_path, "voxel_ground_orbit_video.mp4"),
                default_cam,
                lid_id=ground_id,
            )
    barrier(ddp)

    del scene_mesh
    if mesh_renderer is not None:
        del mesh_renderer
    torch.cuda.empty_cache()

    new_center = None
    new_scale = 1.0
    if opt.recenter_cam:
        with torch.no_grad():
            total_v = []
            for obj_name in dynamic_objs:
                cur_v = saved_meshes[obj_name].v
                deformed_v = dynamic_scene.get_obj_deform(cur_v, obj_name)[1]
                total_v.append(deformed_v)
            total_v = torch.cat(total_v, dim=0)
            vmin, vmax = torch.min(total_v, dim=0).values, torch.max(total_v, dim=0).values
            new_center = ((vmax + vmin) / 2).detach().cpu().numpy()
            new_scale = 1.0
            if ddp.is_main:
                print(f"[CHORD] Recentered camera target: {new_center}")

    render_img_shape = SIZE_CONFIGS[args.size]
    dataset.image_width = render_img_shape[0]
    dataset.image_height = render_img_shape[1]
    cp_orbit_path = os.path.join(dataset.model_path, "orbit_videos")
    cp_deform_path = os.path.join(dataset.model_path, "eval_videos")
    if ddp.is_main:
        os.makedirs(cp_orbit_path, exist_ok=True)
        os.makedirs(cp_deform_path, exist_ok=True)

    default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy)
    if ddp.is_main:
        save_cp_orbit(
            dataset,
            opt,
            dynamic_scene,
            os.path.join(cp_orbit_path, "init_video.mp4"),
            default_cam,
        )
    barrier(ddp)

    if ddp.is_main:
        logging.info(f"Input prompt: {args.prompt}")
        logging.info("Creating Wan pipeline.")

    if "ti2v" in args.task:
        wan_guide = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=ddp.local_rank,
            rank=ddp.rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            init_on_cpu=True,
        )
    else:
        wan_guide = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=ddp.local_rank,
            rank=ddp.rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=True,
            enable_mmgp=opt.enable_mmgp,
            mmgp_profile=opt.mmgp_profile,
            mmgp_transformer_budget=opt.mmgp_transformer_budget,
        )

    if opt.split_sds_backward and not hasattr(wan_guide, "compute_sds_targets"):
        raise ValueError("--split_sds_backward batch target generation is currently supported only for Wan I2V.")

    wan_guide.sds_setup(
        args.prompt,
        frame_num=opt.frame_num,
        size=SIZE_CONFIGS[args.size],
        unet_to_device="cuda",
        resample_timestep=opt.resample_timestep,
        sds_iterations=opt.iterations,
        n_prompt=opt.n_prompt,
        del_decoder=True,
    )
    if opt.use_tiny_vae:
        wan_guide.load_tiny_vae(opt.tiny_vae_path)
    dataset.image_height = wan_guide.h
    dataset.image_width = wan_guide.w
    sds_resolution_scale, temporal_resolution_scale = set_resolution_loss_scales(
        opt,
        wan_guide,
        dataset.image_width,
        dataset.image_height,
    )
    if ddp.is_main:
        print(f"[CHORD] Wan render size: {dataset.image_width}x{dataset.image_height}")
        if opt.log_run_time:
            print(
                "[CHORD] Resolution loss scales: "
                f"sds={sds_resolution_scale:.4f}, temporal={temporal_resolution_scale:.4f}"
            )

    default_cam = OrbitCamera(dataset.image_width, dataset.image_height, r=opt.cam_radius, fovy=dataset.fovy)
    ref_viewmat, ref_opencv_K = build_view(
        dataset,
        default_cam,
        opt.ref_cam_radius * new_scale,
        (opt.elev_l + opt.elev_r) / 2.0,
        opt.ref_azim,
        look_at=new_center,
    )

    progress_bar = tqdm(range(opt.iterations), desc="Training progress") if ddp.is_main else None
    sample_arap_k_ratio = {}
    added_acp = False
    back_iter_flg = False
    stored_arap_coninfo = {}
    save_look_at = np.array([0.0, opt.cam_height, 0.0])
    if opt.recenter_cam:
        save_look_at = new_center.copy()
        save_look_at[1] += opt.cam_height

    for iteration in range(1, opt.iterations + 1):
        if iteration == opt.back_iter and not back_iter_flg:
            if ddp.is_main:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    save_cp_deform(
                        ref_viewmat,
                        ref_opencv_K,
                        dataset,
                        dynamic_scene,
                        os.path.join(cp_deform_path, f"before_prog_deform_video_{iteration}.mp4"),
                        scaling_modifier=0.5,
                    )
                print(f"[CHORD] Assigning deformation up to time {opt.prev_frame_number}")
            barrier(ddp)
            dynamic_scene.assign_deform(opt.prev_frame_number, dynamic_objs)
            broadcast_scene_parameters(dynamic_scene, dynamic_objs, ddp)
            if ddp.is_main:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    save_cp_deform(
                        ref_viewmat,
                        ref_opencv_K,
                        dataset,
                        dynamic_scene,
                        os.path.join(cp_deform_path, f"after_prog_deform_video_{iteration}.mp4"),
                    scaling_modifier=0.5,
                )
            back_iter_flg = True
            barrier(ddp)

        if iteration == opt.add_cp_layer_iter:
            with torch.no_grad():
                if ddp.is_main and len(args.add_cp_objs) > 0:
                    torch.cuda.empty_cache()
                    save_cp_deform(
                        ref_viewmat,
                        ref_opencv_K,
                        dataset,
                        dynamic_scene,
                        os.path.join(cp_deform_path, f"acp_before_video_{iteration}.mp4"),
                        scaling_modifier=0.5,
                    )
                barrier(ddp)
                for obj_name in args.add_cp_objs:
                    if ddp.is_main:
                        print(f"[CHORD] Adding additional control-point layer for {obj_name}")
                    new_voxel_xyzs, sample_ratio = init_distributed_additional_cp(
                        dynamic_scene,
                        obj_name,
                        opt,
                        saved_meshes,
                        mem_voxel_size,
                        mem_voxel_xyzs[obj_name],
                        iteration,
                        ddp,
                    )
                    sample_arap_k_ratio[obj_name] = sample_ratio
                    mem_voxel_xyzs[obj_name] = new_voxel_xyzs
                    added_acp = True
                stored_arap_coninfo.clear()
                broadcast_scene_parameters(dynamic_scene, dynamic_objs, ddp)

                if ddp.is_main and len(args.add_cp_objs) > 0:
                    torch.cuda.empty_cache()
                    save_cp_deform(
                        ref_viewmat,
                        ref_opencv_K,
                        dataset,
                        dynamic_scene,
                        os.path.join(cp_deform_path, f"acp_after_video_{iteration}.mp4"),
                        scaling_modifier=0.5,
                    )
                    render_voxel_grid(
                        dataset,
                        opt,
                        mem_voxel_xyzs,
                        os.path.join(dataset.model_path, "mem_voxel_xyzs_after.mp4"),
                        default_cam,
                        lid_id=None,
                    )
                    save_cp_orbit(
                        dataset,
                        opt,
                        dynamic_scene,
                        os.path.join(cp_orbit_path, "acp_orbit_video_after.mp4"),
                        default_cam,
                        additional=True,
                    )
                barrier(ddp)

        if iteration == opt.convert_cp_independent_iter:
            with torch.no_grad():
                for obj_name in dynamic_objs:
                    dynamic_scene.convert_cp_to_independent(obj_name)
                    if ddp.is_main:
                        print(f"[CHORD] Converting base control points to independent motion for {obj_name}")
                    if iteration >= opt.add_cp_layer_iter and obj_name in args.add_cp_objs:
                        dynamic_scene.convert_additional_cp_to_independent(obj_name)
                        if ddp.is_main:
                            print(f"[CHORD] Converting additional control points to independent motion for {obj_name}")
                broadcast_scene_parameters(dynamic_scene, dynamic_objs, ddp)

        dynamic_scene.update_learning_rate(iteration)
        batch_viewmats, batch_Ks = build_local_batch(
            args,
            dataset,
            opt,
            default_cam,
            new_center,
            new_scale,
        )

        mem_reg_loss = backward_regularization(
            args,
            dataset,
            opt,
            dynamic_scene,
            dynamic_objs,
            mem_voxel_xyzs,
            ground_levels,
            batch_viewmats,
            batch_Ks,
            iteration,
            added_acp,
            sample_arap_k_ratio,
            stored_arap_coninfo,
            ddp,
        )

        step_ratio = min(1, iteration / opt.iterations)
        cur_cfg_scale = opt.init_cfg_scale + (opt.last_cfg_scale - opt.init_cfg_scale) * step_ratio
        mem_sds_loss = backward_sds(
            dataset,
            opt,
            dynamic_scene,
            wan_guide,
            batch_viewmats,
            batch_Ks,
            step_ratio,
            cur_cfg_scale,
        )

        sync_gradients(dynamic_scene, dynamic_objs, ddp)
        reduced_reg_loss = mean_metric(mem_reg_loss, ddp)
        reduced_sds_loss = mean_metric(mem_sds_loss, ddp)

        with torch.no_grad():
            dynamic_scene.optim_step()
            dynamic_scene.optim_zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            should_save = (
                iteration % opt.save_interval == 0
                or iteration == 1
                or iteration == 5
                or iteration == int(opt.iterations * 2 // 3)
            )
            if should_save:
                torch.cuda.empty_cache()
                save_iteration_outputs(
                    iteration,
                    args,
                    dataset,
                    opt,
                    dynamic_scene,
                    default_cam,
                    ref_viewmat,
                    ref_opencv_K,
                    cp_deform_path,
                    added_acp,
                    save_look_at,
                    ddp,
                )
                barrier(ddp)
                if iteration == int(opt.iterations * 2 // 3) and opt.early_stop:
                    return

        if ddp.is_main:
            progress_bar.set_postfix(
                {
                    "regloss": f"{reduced_reg_loss:.4f}",
                    "sdsloss": f"{reduced_sds_loss:.2f}",
                    "cfg": f"{cur_cfg_scale:.2f}",
                }
            )
            progress_bar.update(1)

    if ddp.is_main:
        progress_bar.close()


if __name__ == "__main__":
    args, dataset, opt = _parse_args()
    ddp_info = init_distributed(args)
    if ddp_info.is_main:
        print(f"[CHORD] Setting CUDA device: {ddp_info.local_rank}")
        if args.seed >= 0:
            print(f"[CHORD] Base random seed: {args.seed}")
        if opt.resample_timestep:
            print("[CHORD] Resampling SDS timesteps")
        else:
            print("[CHORD] Using fixed SDS timesteps")
        print(f"[CHORD] Per-GPU batch size: {opt.batch_size}")
        print(f"[CHORD] World size: {ddp_info.world_size}")
    try:
        generate_distributed(args, dataset, opt, ddp_info)
    finally:
        cleanup_distributed(ddp_info)
