# Memory-Constrained Training

This guide explains how to reduce CHORD training memory when the default command does not fit on your GPU. CHORD is expected to work on a **24 GB** GPU

## Split SDS Backward

The first memory-saving option is split SDS backward, which lowers peak memory by splitting parts of SDS gradient calculation and backpropagation. With `--split_sds_backward --split_render_backward --split_render_chunk_size 21` added, CHORD is expected to fit on a **48 GB** GPU.

```bash
python train.py --ex_name <experiment_name> --mesh_source_path data/<scene_name> --model trained/<scene_name> --prompt "<prompt>" --split_sds_backward --split_render_backward --split_render_chunk_size 21 --ckpt_dir ./Wan2.2-I2V-A14B --task i2v-A14B --size "832*480" --n_prompt "." --use_tiny_vae --early_stop --init_voxel_size 0.015 --lambda_ground 0.0 --batch_size 4 --lambda_arap 2.4 --invert_bg_prob -0.5 --azim_l 0.0 --azim_r 360.0 --ref_azim 60.0 --elev_l -10.0 --elev_r 40.0 --cam_radius 2.0 --resample_timestep --cp_num 60 --lambda_dis_time 2.0 --last_cfg_scale 12.0 --init_cfg_scale 25.0 --frame_num 41 --save_interval 500 --iterations 3000
```

`--split_sds_backward` first renders and encodes videos without gradients, then computes detached SDS gradients. CHORD then re-renders and re-encodes each video with gradients enabled, and backpropagates using the precomputed SDS gradients. This reduces peak memory usage because DiT inference no longer needs to keep the Gaussian rendering graph or the VAE/Tiny-VAE encoding graph in memory.

`--split_render_backward` creates a gradient checkpoint at the rendered-video tensor. CHORD first backprops the VAE/Tiny VAE loss to the rendered video, then re-renders the Gaussian video and applies that video gradient, so VAE/Tiny VAE backward does not keep the Gaussian render graph alive.

`--split_render_chunk_size` controls how many frames are re-rendered and backpropped at once in the render-backward step. The default `-1` disables chunking. Smaller positive values use less memory but are slower. For the default `--frame_num 41`, `21` is a practical half-video chunk size.

## Use MMGP

On GPUs where even the full active Wan DiT cannot stay in VRAM during training, use [MMGP](https://github.com/deepbeepmeep/mmgp) residency management. MMGP keeps the Wan DiT weights in system RAM and moves the needed parts between CPU and GPU during DiT inference. This reduces peak VRAM, but it is slower because each SDS target step pays extra CPU/GPU transfer cost.

Use this path explicitly with `--enable_mmgp`, and combine it with split backward. With `--enable_mmgp` plus split backward, CHORD is expected to work on a **24 GB** GPU, assuming enough system RAM for CPU/GPU residency management:

```bash
python train.py --ex_name <experiment_name> --mesh_source_path data/<scene_name> --model trained/<scene_name> --prompt "<prompt>" --enable_mmgp --split_sds_backward --split_render_backward --split_render_chunk_size 21 --ckpt_dir ./Wan2.2-I2V-A14B --task i2v-A14B --size "832*480" --n_prompt "." --use_tiny_vae --early_stop --init_voxel_size 0.015 --lambda_ground 0.0 --batch_size 4 --lambda_arap 2.4 --invert_bg_prob -0.5 --azim_l 0.0 --azim_r 360.0 --ref_azim 60.0 --elev_l -10.0 --elev_r 40.0 --cam_radius 2.0 --resample_timestep --cp_num 60 --lambda_dis_time 2.0 --last_cfg_scale 12.0 --init_cfg_scale 25.0 --frame_num 41 --save_interval 500 --iterations 3000
```


`--enable_mmgp` enables MMGP for Wan DiT residency. Without this flag, CHORD uses the normal Wan loading path.

`--mmgp_profile` selects the MMGP residency profile. The default `4.0` is the low-RAM/low-VRAM profile used by Wan2GP.

`--mmgp_transformer_budget` sets the MMGP transformer VRAM budget in MB. The default `100` is conservative; increasing it can reduce transfer overhead if your GPU has spare memory.


## Lower The Resolution

The next fallback is lowering `--size`. This reduces the rendered video resolution used by Wan SDS, which lowers memory in rendering, VAE/Tiny VAE encoding, and DiT inference. For I2V runs, useful lower-resolution choices include `576*320` and `416*240`.

```bash
python train.py --ex_name <experiment_name> --mesh_source_path data/<scene_name> --model trained/<scene_name> --prompt "<prompt>" --ckpt_dir ./Wan2.2-I2V-A14B --task i2v-A14B --size "576*320" --n_prompt "." --use_tiny_vae --early_stop --init_voxel_size 0.015 --lambda_ground 0.0 --batch_size 4 --lambda_arap 2.4 --invert_bg_prob -0.5 --azim_l 0.0 --azim_r 360.0 --ref_azim 60.0 --elev_l -10.0 --elev_r 40.0 --cam_radius 2.0 --resample_timestep --cp_num 60 --lambda_dis_time 2.0 --last_cfg_scale 12.0 --init_cfg_scale 25.0 --frame_num 41 --save_interval 500 --iterations 3000
```

CHORD automatically rescales the SDS loss and temporal regularization loss relative to the release `832*480` setting when you lower `--size`. This keeps the loss magnitudes closer to the default command.
