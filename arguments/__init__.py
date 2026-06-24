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

from argparse import ArgumentParser


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._model_path = ""
        self._white_background = False
        self.mesh_source_path = "./data/tiger/a_realistic_tiger__0118134023_texture.obj"
        self.mesh_config_path = "arguments/configs/text.yaml"
        self.image_width = 624
        self.image_height = 624
        self.fovy = 49.1
        self.is_marbles = False
        super().__init__(parser, "Loading Parameters", sentinel)

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 50_000
        self.load_from_checkpoint = False
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.train_with_depth = False
        self.lambda_scale = 0.0
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.001
        self.max_scale = -1.0
        self.invert_bg_prob = 0.5
        self.opacity_reduce_interval = 500
        self.opacity_reduce_until_iter = 30000
        self.elev_l = -60.0
        self.elev_r = 60.0
        self.azim_l = -180.0
        self.azim_r = 180.0
        self.cam_radius = 2.0
        self.use_light = False
        self.test = False
        self.static_id = 1
        self.move_cam_radius = -1.0
        self.light_bg = False
        self.train_by_parts = False
        self.zoom_in_train = False
        self.penetrate_voxel_size = 0.012
        self.near_cam_radius = 0.5
        self.zoom_in_iter = 2000
        self.cur_train = "obj_0"
        self.cam_fixed_on_obj = 1.0
        self.self_norm = False
        super().__init__(parser, "Optimization Parameters")

class SDSOptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 3_000

        self.deform_lr_init = 0.006
        self.deform_lr_final = 0.00006
        self.deform_lr_delay_mult = 0.01
        self.deform_lr_max_steps = self.iterations

        self.cp_radius_lr_init = 0.006
        self.cp_radius_lr_final = 0.00006
        self.cp_radius_lr_delay_mult = 0.01
        self.cp_radius_lr_max_steps = self.iterations

        self.cp_rotation_lr_init = 0.003
        self.cp_rotation_lr_final = 0.00003
        self.cp_rotation_lr_delay_mult = 0.01
        self.cp_rotation_lr_max_steps = self.iterations

        # LR for additional layers
        self.c_deform_lr_init = 0.002
        self.c_deform_lr_final = 0.00002
        self.c_deform_lr_delay_mult = 0.01
        self.c_deform_lr_max_steps = self.iterations

        self.c_cp_radius_lr_init = 0.006
        self.c_cp_radius_lr_final = 0.00006
        self.c_cp_radius_lr_delay_mult = 0.01
        self.c_cp_radius_lr_max_steps = self.iterations

        self.c_cp_rotation_lr_init = 0.003
        self.c_cp_rotation_lr_final = 0.00003
        self.c_cp_rotation_lr_delay_mult = 0.01
        self.c_cp_rotation_lr_max_steps = self.iterations

        self.frame_num = 41
        self.batch_size = 4

        self.lambda_arap = 2.4
        self.arap_sample_num = 1024
        self.ex_name = "default"

        self.lambda_dis_time = 2.0

        self.cp_num = 60

        self.detach_radius = False

        self.invert_bg_prob = -1.0

        self.elev_l = -10.0
        self.elev_r = 40.0

        self.azim_l = 0.0
        self.azim_r = 360.0

        self.save_interval = 500

        self.init_cfg_scale = 25.0
        self.last_cfg_scale = 12.0

        self.save_fps = 10

        self.resample_timestep = False

        self.n_cp_num = 3

        self.cam_radius = 1.8
        self.cam_height = 0.0

        self.ref_azim = 60.0
        self.ref_cam_radius = 2.0

        self.obj_num = -1
        self.static_id = 1

        self.lambda_dis_landmarks = [6.0, 6.0, 6.0, 4.0, 1.0]
        self.lambda_arap_landmarks = [1200.0, 800.0, 400.0, 200.0, 50.0]
        self.landmark_steps = [0,  500, 1000, 1500, 3000]

        self.early_stop = False

        self.n_prompt = "."
        self.use_tiny_vae = False
        self.tiny_vae_path = "tiny_vae/lighttaew2_1.pth"

        self.split_sds_backward = False
        self.enable_mmgp = False
        self.mmgp_profile = 4.0
        self.mmgp_transformer_budget = 100
        self.split_render_backward = False
        self.split_render_chunk_size = -1

        self.lambda_ground = 0.0

        self.replace_voxel_with_orig = False
        self.init_voxel_size = 0.015

        self.fix_ground = False

        self.add_cp_layer_iter = 300
        self.add_cp_num = 1000
        self.add_cp_voxel_scale = 2.5
        self.add_cp_detach_first = False
        self.mult_rot_way = 0

        self.br_range = 0.75

        self.back_iter = 100
        self.prev_frame_number = 29

        self.recenter_cam = False

        self.convert_cp_independent_iter = 100000

        self.log_run_time = False

        self.surface_target_num = 8000

        self.scheduler_type = "cosine"
        self.warmup_steps = 100
        self.use_lr_scheduler = False

        super().__init__(parser, "Optimization Parameters")
