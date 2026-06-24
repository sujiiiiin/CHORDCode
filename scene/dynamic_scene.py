import os

import torch


class DynamicGaussianScene:
    def __init__(self, total_time: int):
        self.total_time = total_time
        self.dynamic_gaussians = {}

    def add_gaussians(self, gaussians, is_static=False, name=""):
        self.dynamic_gaussians[name] = (gaussians, is_static)

    def query_static(self, obj_name):
        entry = self.dynamic_gaussians.get(obj_name)
        if entry is None:
            raise KeyError(f"Unknown object '{obj_name}'.")
        return entry[1]

    def get_scaling(self, time=None):
        scalings = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            scalings.append(gaussians.get_scaling if is_static else gaussians.get_scaling(time))
        return torch.cat(scalings, dim=0)

    def optim_step(self):
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                gaussians.optimizer.step()

    def optim_zero_grad(self, set_to_none=True):
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                gaussians.optimizer.zero_grad(set_to_none=set_to_none)

    def get_xyz_rotation(
        self,
        time,
        detach_node_radius=True,
    ):
        xyz_chunks = []
        rotation_chunks = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if is_static:
                xyz = gaussians.get_xyz.detach()
                rotation = gaussians.get_rotation.detach()
            else:
                xyz, rotation = gaussians.get_xyz_rotation(
                    time, detach_node_radius
                )
            xyz_chunks.append(xyz)
            rotation_chunks.append(rotation)
        return torch.cat(xyz_chunks, dim=0), torch.cat(rotation_chunks, dim=0)

    def get_xyz_rotation_whole(
        self,
        detach_node_radius=True,
    ):
        xyz_chunks = []
        rotation_chunks = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if is_static:
                xyz = gaussians.get_xyz.detach().unsqueeze(0).repeat(self.total_time, 1, 1)
                rotation = gaussians.get_rotation.detach().unsqueeze(0).repeat(
                    self.total_time, 1, 1
                )
            else:
                xyz, rotation = gaussians.get_xyz_rotation_whole(
                    detach_node_radius
                )
            xyz_chunks.append(xyz)
            rotation_chunks.append(rotation)
        return torch.cat(xyz_chunks, dim=1), torch.cat(rotation_chunks, dim=1)

    def get_xyz_rotation_range(
        self,
        start_time,
        end_time,
        detach_node_radius=True,
    ):
        chunk_time = end_time - start_time
        xyz_chunks = []
        rotation_chunks = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if is_static:
                xyz = gaussians.get_xyz.detach().unsqueeze(0).repeat(chunk_time, 1, 1)
                rotation = gaussians.get_rotation.detach().unsqueeze(0).repeat(
                    chunk_time, 1, 1
                )
            else:
                xyz, rotation = gaussians.get_xyz_rotation_range(
                    start_time,
                    end_time,
                    detach_node_radius,
                )
            xyz_chunks.append(xyz)
            rotation_chunks.append(rotation)
        return torch.cat(xyz_chunks, dim=1), torch.cat(rotation_chunks, dim=1)

    def get_obj_deform(self, xyz, obj_name, xyz_id=None):
        gaussians, _ = self.dynamic_gaussians[obj_name]
        if xyz_id is None:
            return gaussians.query_xyz(xyz)
        return gaussians.query_xyz(xyz, xyz_id)

    def get_obj_deform_whole(self, xyz, obj_name):
        gaussians, _ = self.dynamic_gaussians[obj_name]
        return gaussians.query_xyz_time_whole(xyz)

    def init_object_additional_cp(
        self,
        obj_name,
        training_args,
        init_cp_num,
        detach_first_layer=False,
        init_mesh=None,
        init_voxel_size=0.015,
        mult_rot_way=0,
        target_num=8000,
    ):
        gaussians, is_static = self.dynamic_gaussians[obj_name]
        if is_static:
            raise ValueError(f"Cannot add control points to static object '{obj_name}'.")
        voxel_xyzs = gaussians.init_additional_control_points(
            init_cp_num,
            detach_first_layer=detach_first_layer,
            init_mesh=init_mesh,
            init_voxel_size=init_voxel_size,
            mult_rot_way=mult_rot_way,
            target_num=target_num
        )
        gaussians.append_additional_cp_to_optimizer(training_args)
        return voxel_xyzs

    def convert_cp_to_independent(self, obj_name):
        gaussians, _ = self.dynamic_gaussians[obj_name]
        gaussians.convert_cp_to_independent()

    def convert_additional_cp_to_independent(self, obj_name):
        gaussians, _ = self.dynamic_gaussians[obj_name]
        gaussians.convert_additional_cp_to_independent()

    def get_cp_position(self, time):
        cp_positions = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                cp_positions.append(gaussians.get_cp_position(time))
        return torch.cat(cp_positions, dim=0)

    def get_cp_rotation(self, time):
        cp_rotations = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                cp_rotations.append(gaussians.get_cp_rotation(time))
        return torch.cat(cp_rotations, dim=0)

    def get_cp_scaling(self, time=None):
        _ = time
        cp_scalings = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                cp_scalings.append(gaussians.get_cp_scaling())
        return torch.cat(cp_scalings, dim=0)

    def get_additional_cp_position(self, time):
        cp_positions = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static and gaussians.c_cp_deform is not None:
                cp_positions.append(gaussians.get_additional_cp_position(time))
        if not cp_positions:
            raise ValueError("No additional control points are available in the scene.")
        return torch.cat(cp_positions, dim=0)

    def get_additional_cp_rotation(self, time):
        cp_rotations = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static and gaussians.c_cp_deform is not None:
                cp_rotations.append(gaussians.get_additional_cp_rotation(time))
        if not cp_rotations:
            raise ValueError("No additional control points are available in the scene.")
        return torch.cat(cp_rotations, dim=0)

    def get_additional_cp_scaling(self, time=None):
        _ = time
        cp_scalings = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static and gaussians.c_cp_deform is not None:
                cp_scalings.append(gaussians.get_additional_cp_scaling())
        if not cp_scalings:
            raise ValueError("No additional control points are available in the scene.")
        return torch.cat(cp_scalings, dim=0)

    def get_features(self, time=None):
        features = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            features.append(gaussians.get_features if is_static else gaussians.get_features(time))
        return torch.cat(features, dim=0)

    def get_opacity(self, time=None):
        opacities = []
        for gaussians, is_static in self.dynamic_gaussians.values():
            opacities.append(gaussians.get_opacity if is_static else gaussians.get_opacity(time))
        return torch.cat(opacities, dim=0)

    def training_setup(self, training_args):
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                gaussians.training_setup(training_args)

    def update_learning_rate(self, iteration):
        for gaussians, is_static in self.dynamic_gaussians.values():
            if not is_static:
                gaussians.update_learning_rate(iteration)

    def save_pth(self, path):
        os.makedirs(path, exist_ok=True)
        for name, (gaussians, is_static) in self.dynamic_gaussians.items():
            if not is_static:
                gaussians.save_pth(os.path.join(path, f"{name}.pth"))

    def assign_deform(self, max_time, assign_names=("obj_0",)):
        for name, (gaussians, is_static) in self.dynamic_gaussians.items():
            if not is_static and name in assign_names:
                gaussians.assign_deform(max_time)
