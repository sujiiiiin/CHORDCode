import torch
import torch.distributed as dist
import torch.nn as nn

from utils.arap_utils import cal_connectivity_from_points
from utils.runtime_utils import barrier, broadcast_object


def make_bit_motion(payload):
    from scene.dynamic_gaussian_model import BitMotionBase

    motion = BitMotionBase(payload["total_time"], payload["cp_num"])
    motion.cp_deform = nn.Parameter(
        payload["cp_deform"].clone().cuda().requires_grad_(True)
    )
    motion.converted_to_independent = payload["converted_to_independent"]
    return motion


def extract_bit_motion(motion):
    return {
        "total_time": motion.total_time,
        "cp_num": motion.cp_num,
        "cp_deform": motion.cp_deform.detach().cpu(),
        "converted_to_independent": bool(motion.converted_to_independent),
    }


def extract_dynamic_payload(gaussians):
    c_payload = None
    if gaussians.c_cp_deform is not None:
        c_payload = {
            "c_cp_center": gaussians.c_cp_center.detach().cpu(),
            "c_cp_radius": gaussians.c_cp_radius.detach().cpu(),
            "c_cp_rotation": gaussians.c_cp_rotation.detach().cpu(),
            "c_cp_deform": extract_bit_motion(gaussians.c_cp_deform),
            "detach_first_layer": bool(gaussians.detach_first_layer),
            "mult_rot_way": int(gaussians.mult_rot_way),
        }

    return {
        "br_range": float(gaussians.br_range),
        "cp_center": gaussians.cp_center.detach().cpu(),
        "cp_radius": gaussians.cp_radius.detach().cpu(),
        "cp_rotation": gaussians.cp_rotation.detach().cpu(),
        "cp_deform": extract_bit_motion(gaussians.cp_deform),
        "voxel_xyz": gaussians.voxel_xyz.detach().cpu(),
        "mem_init_voxel_size": gaussians.mem_init_voxel_size,
        "c_payload": c_payload,
    }


def install_dynamic_payload(gaussians, payload):
    gaussians.br_range = payload["br_range"]
    gaussians.cp_center = payload["cp_center"].clone().cuda().detach()
    gaussians.cp_radius = nn.Parameter(
        payload["cp_radius"].clone().cuda().requires_grad_(True)
    )
    gaussians.cp_rotation = nn.Parameter(
        payload["cp_rotation"].clone().cuda().requires_grad_(True)
    )
    gaussians.cp_deform = make_bit_motion(payload["cp_deform"])
    gaussians.voxel_xyz = payload["voxel_xyz"].clone().cuda()
    gaussians.voxel_grid = None
    gaussians.mem_init_voxel_size = payload["mem_init_voxel_size"]
    install_additional_payload(gaussians, payload["c_payload"])


def extract_additional_payload(gaussians, voxel_xyz):
    return {
        "voxel_xyz": voxel_xyz.detach().cpu(),
        "c_payload": {
            "c_cp_center": gaussians.c_cp_center.detach().cpu(),
            "c_cp_radius": gaussians.c_cp_radius.detach().cpu(),
            "c_cp_rotation": gaussians.c_cp_rotation.detach().cpu(),
            "c_cp_deform": extract_bit_motion(gaussians.c_cp_deform),
            "detach_first_layer": bool(gaussians.detach_first_layer),
            "mult_rot_way": int(gaussians.mult_rot_way),
        },
    }


def install_additional_payload(gaussians, payload):
    if payload is None:
        gaussians.c_cp_deform = None
        gaussians.c_cp_radius = None
        gaussians.c_cp_center = None
        gaussians.c_cp_rotation = None
        gaussians.detach_first_layer = False
        return

    gaussians.c_cp_center = payload["c_cp_center"].clone().cuda().detach()
    gaussians.c_cp_radius = nn.Parameter(
        payload["c_cp_radius"].clone().cuda().requires_grad_(True)
    )
    gaussians.c_cp_rotation = nn.Parameter(
        payload["c_cp_rotation"].clone().cuda().requires_grad_(True)
    )
    gaussians.c_cp_deform = make_bit_motion(payload["c_cp_deform"])
    gaussians.detach_first_layer = payload["detach_first_layer"]
    gaussians.mult_rot_way = payload["mult_rot_way"]


def broadcast_tensor_data(tensor, ddp, src=0):
    if not ddp.distributed or tensor is None:
        return
    target = tensor.data if isinstance(tensor, nn.Parameter) else tensor
    dist.broadcast(target, src=src)


def broadcast_scene_parameters(dynamic_scene, dynamic_objs, ddp):
    if not ddp.distributed:
        return
    for obj_name in dynamic_objs:
        gaussians, is_static = dynamic_scene.dynamic_gaussians[obj_name]
        if is_static:
            continue
        broadcast_tensor_data(gaussians.cp_center, ddp)
        broadcast_tensor_data(gaussians.cp_radius, ddp)
        broadcast_tensor_data(gaussians.cp_rotation, ddp)
        broadcast_tensor_data(gaussians.cp_deform.cp_deform, ddp)
        if gaussians.c_cp_deform is not None:
            broadcast_tensor_data(gaussians.c_cp_center, ddp)
            broadcast_tensor_data(gaussians.c_cp_radius, ddp)
            broadcast_tensor_data(gaussians.c_cp_rotation, ddp)
            broadcast_tensor_data(gaussians.c_cp_deform.cp_deform, ddp)


def sync_gradients(dynamic_scene, dynamic_objs, ddp):
    if not ddp.distributed:
        return
    for obj_name in dynamic_objs:
        gaussians, is_static = dynamic_scene.dynamic_gaussians[obj_name]
        if is_static:
            continue
        for group in gaussians.optimizer.param_groups:
            for param in group["params"]:
                has_grad = torch.tensor(
                    [param.grad is not None],
                    dtype=torch.int32,
                    device=ddp.device,
                )
                dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
                if has_grad.item() == 0:
                    param.grad = None
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(ddp.world_size)


def build_synced_arap_coninfo(means3d, ddp):
    nodes_t = means3d.permute(1, 0, 2)
    hyper_nodes = nodes_t[:, 0]
    payload = None
    if not ddp.distributed or ddp.is_main:
        ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)
        payload = {
            "ii": ii.detach().cpu(),
            "jj": jj.detach().cpu(),
            "nn": nn.detach().cpu(),
            "weight": weight.detach().cpu(),
        }
    if ddp.distributed:
        payload = broadcast_object(payload, ddp)
    return {key: value.to(hyper_nodes.device) for key, value in payload.items()}


def init_distributed_additional_cp(
    dynamic_scene,
    obj_name,
    opt,
    saved_meshes,
    mem_voxel_size,
    old_voxel_xyz,
    iteration,
    ddp,
):
    gaussians, _ = dynamic_scene.dynamic_gaussians[obj_name]
    payload = None
    if ddp.is_main:
        new_voxel_xyzs = gaussians.init_additional_control_points(
            opt.add_cp_num,
            detach_first_layer=opt.add_cp_detach_first,
            init_mesh=saved_meshes[obj_name],
            init_voxel_size=mem_voxel_size[obj_name] / opt.add_cp_voxel_scale,
            mult_rot_way=opt.mult_rot_way,
            target_num=opt.surface_target_num,
        )
        payload = extract_additional_payload(gaussians, new_voxel_xyzs)

    barrier(ddp)
    payload = broadcast_object(payload, ddp)
    if not ddp.is_main:
        install_additional_payload(gaussians, payload["c_payload"])

    gaussians.append_additional_cp_to_optimizer(opt)
    new_voxel_xyzs = payload["voxel_xyz"].clone().cuda()
    sample_ratio = new_voxel_xyzs.shape[0] / old_voxel_xyz.shape[0]
    if iteration > opt.convert_cp_independent_iter:
        gaussians.convert_additional_cp_to_independent()
    return new_voxel_xyzs, sample_ratio
