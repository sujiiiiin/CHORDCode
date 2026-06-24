import numpy as np
import torch

from utils.arap_utils import arap_loss, calc_temp_loss, landmark_interpolate
from utils.ddp_utils import build_synced_arap_coninfo
from utils.general_utils import sample_from_segments
from utils.render_utils import (
    backward_sds_render_chunks,
    build_view,
    render_sds_training_video,
)
from utils.resolution_loss_utils import (
    get_sds_resolution_scale,
    get_temporal_resolution_scale,
)


def _start_timing(enabled):
    if not enabled:
        return None
    event_pair = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    event_pair[0].record()
    return event_pair


def _finish_timing(event_pair, label):
    if event_pair is None:
        return
    start_event, end_event = event_pair
    end_event.record()
    torch.cuda.synchronize()
    print(f"[Timing] {label}: {start_event.elapsed_time(end_event) / 1000.0}")


def build_local_batch(args, dataset, opt, default_cam, new_center, new_scale):
    batch_viewmats = []
    batch_ks = []
    for _ in range(opt.batch_size):
        if len(args.azim_segments) > 0:
            azim = sample_from_segments(args.azim_segments)
        else:
            azim = np.random.uniform(opt.azim_l, opt.azim_r)
        if len(args.elev_segments) > 0:
            elev = sample_from_segments(args.elev_segments)
        else:
            elev = np.random.uniform(opt.elev_l, opt.elev_r)
        look_at = np.array([0.0, 0.0, 0.0])
        cur_cam_radius = opt.cam_radius
        if opt.recenter_cam:
            look_at = new_center.copy()
            cur_cam_radius = cur_cam_radius * new_scale
        look_at[1] += opt.cam_height
        viewmat, opencv_k = build_view(
            dataset,
            default_cam,
            cur_cam_radius,
            elev,
            azim,
            look_at=look_at,
        )
        batch_viewmats.append(viewmat)
        batch_ks.append(opencv_k)
    return batch_viewmats, batch_ks


def backward_regularization(
    args,
    dataset,
    opt,
    dynamic_scene,
    dynamic_objs,
    mem_voxel_xyzs,
    ground_levels,
    batch_viewmats,
    batch_ks,
    iteration,
    added_acp,
    sample_arap_k_ratio,
    stored_arap_coninfo,
    ddp,
):
    mem_reg_loss = torch.tensor(0.0, device="cuda")

    if opt.lambda_ground > 0.0:
        for obj_name in args.gl_objects:
            if obj_name not in dynamic_objs:
                continue
            voxel_xyz = mem_voxel_xyzs[obj_name]
            ground_level = ground_levels[obj_name]
            means3d = dynamic_scene.get_obj_deform(voxel_xyz, obj_name)
            deformed_y = means3d[:, :, 1]
            mask = deformed_y < ground_level
            if mask.any():
                ground_penalty = ((deformed_y[mask] - ground_level) ** 2.0).mean()
                mem_reg_loss = mem_reg_loss + ground_penalty * opt.lambda_ground

    viewmats_tensor = torch.cat(batch_viewmats, dim=0)
    ks_tensor = torch.cat(batch_ks, dim=0)

    cur_lambda_arap = opt.lambda_arap * landmark_interpolate(
        opt.lambda_arap_landmarks,
        opt.landmark_steps,
        iteration,
    )
    cur_lambda_dis_time = opt.lambda_dis_time * landmark_interpolate(
        opt.lambda_dis_landmarks,
        opt.landmark_steps,
        iteration,
    )

    if cur_lambda_dis_time > 0.0:
        time_loss = calc_temp_loss(
            dynamic_scene,
            opt.frame_num,
            viewmats_tensor,
            ks_tensor,
            dataset.image_width,
            dataset.image_height,
        )
        mem_reg_loss = (
            mem_reg_loss
            + time_loss
            * cur_lambda_dis_time
            * get_temporal_resolution_scale(opt)
            / opt.batch_size
        )

    if cur_lambda_arap > 0.0:
        for obj_name in dynamic_objs:
            voxel_xyz = mem_voxel_xyzs[obj_name]
            means3d = dynamic_scene.get_obj_deform_whole(voxel_xyz, obj_name)
            if opt.replace_voxel_with_orig:
                means3d[0] = voxel_xyz
            arap_sample_num = opt.arap_sample_num
            if added_acp and obj_name in args.add_cp_objs:
                arap_sample_num = int(opt.arap_sample_num * sample_arap_k_ratio[obj_name])
            if obj_name in stored_arap_coninfo:
                araploss, coninfo = arap_loss(
                    means3d,
                    sample_num=arap_sample_num,
                    sample_times=opt.batch_size,
                    stored_coninfo=stored_arap_coninfo[obj_name],
                )
            else:
                coninfo = build_synced_arap_coninfo(means3d, ddp)
                araploss, coninfo = arap_loss(
                    means3d,
                    sample_num=arap_sample_num,
                    sample_times=opt.batch_size,
                    stored_coninfo=coninfo,
                )
                stored_arap_coninfo[obj_name] = coninfo
            mem_reg_loss = mem_reg_loss + araploss * cur_lambda_arap / opt.batch_size

    if cur_lambda_arap > 0.0 or cur_lambda_dis_time > 0.0 or opt.lambda_ground > 0.0:
        mem_reg_loss.backward()

    return mem_reg_loss


def backward_sds(
    dataset,
    opt,
    dynamic_scene,
    wan_guide,
    batch_viewmats,
    batch_ks,
    step_ratio,
    cur_cfg_scale,
):
    mem_sds_loss = 0.0

    if opt.split_sds_backward:
        render_specs = []
        for b in range(opt.batch_size):
            bg_color = 1 if np.random.rand() > opt.invert_bg_prob else 0
            render_specs.append((batch_viewmats[b], batch_ks[b], bg_color))

        rendered_videos = []
        with torch.no_grad():
            for cur_viewmat, cur_opencv_k, bg_color in render_specs:
                rendered_videos.append(
                    render_sds_training_video(
                        cur_viewmat,
                        cur_opencv_k,
                        bg_color,
                        dataset,
                        opt,
                        dynamic_scene,
                    )
                )

        sds_targets = wan_guide.compute_sds_targets(
            rendered_videos,
            step_ratio=step_ratio,
            pred_flow=True,
            guidance_scale=cur_cfg_scale,
            log_run_time=opt.log_run_time,
            use_tiny_vae=opt.use_tiny_vae,
        )
        del rendered_videos
        wan_guide.release_active_dit_after_target(log_run_time=opt.log_run_time)

        for (cur_viewmat, cur_opencv_k, bg_color), sds_target in zip(render_specs, sds_targets):
            total_timer = _start_timing(opt.log_run_time)
            if opt.split_render_backward:
                with torch.no_grad():
                    rendered_video_leaf = render_sds_training_video(
                        cur_viewmat,
                        cur_opencv_k,
                        bg_color,
                        dataset,
                        opt,
                        dynamic_scene,
                    ).detach().requires_grad_(True)

                cur_sds_loss = wan_guide.sds_loss_from_target(
                    rendered_video_leaf,
                    sds_target,
                    use_tiny_vae=opt.use_tiny_vae,
                )
                cur_sds_loss = cur_sds_loss * get_sds_resolution_scale(opt) / opt.batch_size
                vae_backward_timer = _start_timing(opt.log_run_time)
                cur_sds_loss.backward()
                _finish_timing(vae_backward_timer, "SDS VAE backward")
                mem_sds_loss += cur_sds_loss.item()
                if rendered_video_leaf.grad is None:
                    raise RuntimeError("Manual split render backward did not receive a video gradient.")
                video_grad = rendered_video_leaf.grad.detach()
                del rendered_video_leaf, cur_sds_loss, sds_target
                sds_target = None
                torch.cuda.empty_cache()

                render_backward_timer = _start_timing(opt.log_run_time)
                backward_sds_render_chunks(
                    cur_viewmat,
                    cur_opencv_k,
                    bg_color,
                    dataset,
                    opt,
                    dynamic_scene,
                    video_grad,
                )
                _finish_timing(render_backward_timer, "SDS render backward")
                del video_grad
            else:
                rendered_video = render_sds_training_video(
                    cur_viewmat,
                    cur_opencv_k,
                    bg_color,
                    dataset,
                    opt,
                    dynamic_scene,
                )
                cur_sds_loss = wan_guide.sds_loss_from_target(
                    rendered_video,
                    sds_target,
                    use_tiny_vae=opt.use_tiny_vae,
                )
                cur_sds_loss = cur_sds_loss * get_sds_resolution_scale(opt) / opt.batch_size
                backward_timer = _start_timing(opt.log_run_time)
                cur_sds_loss.backward()
                _finish_timing(backward_timer, "SDS backward")
                mem_sds_loss += cur_sds_loss.item()
                del rendered_video, cur_sds_loss

            _finish_timing(total_timer, "SDS backward total")
            del sds_target
        del render_specs, sds_targets
        torch.cuda.empty_cache()
        return mem_sds_loss

    for b in range(opt.batch_size):
        bg_color = 1 if np.random.rand() > opt.invert_bg_prob else 0
        rendered_video = render_sds_training_video(
            batch_viewmats[b],
            batch_ks[b],
            bg_color,
            dataset,
            opt,
            dynamic_scene,
        )
        cur_sds_loss = wan_guide.sds_step(
            rendered_video,
            step_ratio,
            pred_flow=True,
            guidance_scale=cur_cfg_scale,
            log_run_time=opt.log_run_time,
            use_tiny_vae=opt.use_tiny_vae,
        )
        cur_sds_loss = cur_sds_loss * get_sds_resolution_scale(opt) / opt.batch_size
        backward_timer = _start_timing(opt.log_run_time)
        cur_sds_loss.backward()
        _finish_timing(backward_timer, "SDS backward")
        mem_sds_loss += cur_sds_loss.item()
        del rendered_video, cur_sds_loss

    return mem_sds_loss
