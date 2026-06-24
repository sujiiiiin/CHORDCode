# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import json
import logging
import math
import os
import random
import time
import types
import gc
from functools import partial

import numpy as np
from scipy.stats import norm
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE


def _logit_normal_pdf(t, mu=0.0, sigma=1.0):
    eps = 1e-6
    t = np.clip(t, eps, 1.0 - eps)
    logit_t = np.log(t / (1.0 - t))
    base_pdf = norm.pdf(logit_t, loc=mu, scale=sigma)
    return base_pdf / (t * (1.0 - t))


def _sample_weighted_logit_normal_timesteps(mu=0.0, sigma=1.0, total_steps=50, grid_size=1_000_000):
    t_grid = np.linspace(1e-6, 1.0 - 1e-6, grid_size)
    x_t = _logit_normal_pdf(t_grid, mu, sigma)
    weights = (t_grid / (1.0 - t_grid)) * x_t
    weights /= np.trapz(weights, t_grid)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    target_cdf_vals = np.linspace(1.0 / total_steps, (total_steps - 1) / total_steps, total_steps - 1)
    return np.interp(target_cdf_vals, cdf, t_grid)


def _w_logit_normal(t, m=0.0, s=1.0):
    logit_t = np.log(t / (1.0 - t))
    coeff = 1.0 / (s * np.sqrt(2.0 * np.pi) * t * (1.0 - t))
    exponent = -((logit_t - m) ** 2) / (2.0 * s ** 2)
    pi_t = coeff * np.exp(exponent)
    return (t / (1.0 - t)) * pi_t


def _timed_cuda_placement(label, fn, log_run_time=False):
    if not log_run_time:
        return fn()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    print(f"[Timing] {label}: {time.perf_counter() - start_time}")
    return result


class WanI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        enable_mmgp=False,
        mmgp_profile=4.0,
        mmgp_transformer_budget=100,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.enable_mmgp = bool(enable_mmgp)
        self.mmgp_profile = float(mmgp_profile)
        self.mmgp_transformer_budget = int(mmgp_transformer_budget)
        self.mmgp_offload_objects = {}
        self.mmgp_loaded_model_name = None

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if self.enable_mmgp and (dit_fsdp or use_sp):
            raise ValueError("--enable_mmgp is not compatible with DiT FSDP or sequence parallelism.")

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        if self.enable_mmgp:
            self.init_on_cpu = True
            self._load_mmgp_offload()

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )

        logging.info("Creating WanModel from %s", checkpoint_dir)
        if self.enable_mmgp:
            self.mmgp_checkpoint_dir = checkpoint_dir
            self.mmgp_checkpoint_subfolders = {
                "low_noise_model": config.low_noise_checkpoint,
                "high_noise_model": config.high_noise_checkpoint,
            }
            self.low_noise_model = None
            self.high_noise_model = None
        else:
            self.low_noise_model = self._load_wan_model(checkpoint_dir, config.low_noise_checkpoint)
            self.low_noise_model = self._configure_model(
                model_name="low_noise_model",
                model=self.low_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
            )

            self.high_noise_model = self._load_wan_model(checkpoint_dir, config.high_noise_checkpoint)
            self.high_noise_model = self._configure_model(
                model_name="high_noise_model",
                model=self.high_noise_model,
                use_sp=use_sp,
                dit_fsdp=dit_fsdp,
                shard_fn=shard_fn,
                convert_model_dtype=convert_model_dtype,
            )

        self.sp_size = get_world_size() if use_sp else 1
        self.sample_neg_prompt = config.sample_neg_prompt

    def _load_mmgp_offload(self):
        try:
            from mmgp import offload
        except ImportError as exc:
            raise ImportError(
                "Using --enable_mmgp requires MMGP. Install it with:\n"
                "  pip install mmgp==3.7.6"
            ) from exc
        self.mmgp_offload = offload

    def _get_mmgp_weight_files(self, model_dir):
        index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            filenames = sorted(set(index.get("weight_map", {}).values()))
        else:
            filenames = sorted(
                filename
                for filename in os.listdir(model_dir)
                if filename.endswith(".safetensors")
            )

        if not filenames:
            raise FileNotFoundError(f"No Wan DiT safetensors found in {model_dir}")
        return [os.path.join(model_dir, filename) for filename in filenames]

    def _load_wan_model(self, checkpoint_dir, subfolder):
        if not self.enable_mmgp:
            return WanModel.from_pretrained(checkpoint_dir, subfolder=subfolder)

        model_dir = os.path.join(checkpoint_dir, subfolder)
        config_path = os.path.join(model_dir, "config.json")
        return self.mmgp_offload.fast_load_transformers_model(
            self._get_mmgp_weight_files(model_dir),
            modelClass=WanModel,
            defaultConfigPath=config_path,
            forcedConfigPath=config_path,
            writable_tensors=False,
            default_dtype=self.param_dtype,
            do_quantize=False,
        )

    def _profile_mmgp_model(self, model_name, model):
        pipe = {"transformer": model}
        kwargs = {
            "profile_no": self.mmgp_profile,
            "quantizeTransformer": False,
            "extraModelsToQuantize": None,
            "budgets": {
                "transformer": self.mmgp_transformer_budget,
                "*": 3000,
            },
            "convertWeightsFloatTo": self.param_dtype,
        }
        if self.mmgp_profile == 4.5:
            kwargs["asyncTransfers"] = False

        self.mmgp_offload_objects[model_name] = self.mmgp_offload.profile(pipe, **kwargs)
        return pipe["transformer"]

    def _configure_mmgp_model(self, model_name, model):
        model.eval().requires_grad_(False)
        model.to(self.param_dtype)
        return self._profile_mmgp_model(model_name, model)

    def _release_mmgp_model(self, model_name):
        if model_name is None:
            return
        offload_object = self.mmgp_offload_objects.pop(model_name, None)
        if offload_object is not None:
            if hasattr(offload_object, "release"):
                offload_object.release()
            else:
                offload_object.unload_all()
        setattr(self, model_name, None)
        if self.mmgp_loaded_model_name == model_name:
            self.mmgp_loaded_model_name = None
        gc.collect()
        torch.cuda.empty_cache()

    def _ensure_mmgp_model(self, model_name):
        model = getattr(self, model_name)
        if model is not None:
            return model

        if self.mmgp_loaded_model_name is not None and self.mmgp_loaded_model_name != model_name:
            self._release_mmgp_model(self.mmgp_loaded_model_name)

        subfolder = self.mmgp_checkpoint_subfolders[model_name]
        model = self._load_wan_model(self.mmgp_checkpoint_dir, subfolder)
        model = self._configure_mmgp_model(model_name, model)
        setattr(self, model_name, model)
        self.mmgp_loaded_model_name = model_name
        return model

    def _configure_model(self, model_name, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype):
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype or self.enable_mmgp:
                model.to(self.param_dtype)
            if self.enable_mmgp:
                model = self._profile_mmgp_model(model_name, model)
            elif not self.init_on_cpu:
                model.to(self.device)

        return model

    def _move_model_to_forward_device(self, model, log_run_time=False):
        if hasattr(model, "prepare_dit_for_forward"):
            return _timed_cuda_placement(
                "DiT onload",
                lambda: model.prepare_dit_for_forward(self.device),
                log_run_time=log_run_time,
            )
        return _timed_cuda_placement(
            "DiT onload",
            lambda: model.to(self.device),
            log_run_time=log_run_time,
        )

    def offload_dit_models(self):
        if self.enable_mmgp:
            for model_name in list(self.mmgp_offload_objects):
                self._release_mmgp_model(model_name)
            torch.cuda.empty_cache()
            return
        self.low_noise_model.to("cpu")
        self.high_noise_model.to("cpu")
        torch.cuda.empty_cache()

    def release_active_dit_after_target(self, log_run_time=False):
        active_model_name = getattr(self, "active_dit_model_name", None)
        if active_model_name is None:
            return
        if self.enable_mmgp:
            offload_object = self.mmgp_offload_objects.get(active_model_name)
            if offload_object is not None:
                _timed_cuda_placement(
                    "MMGP active DiT unload",
                    offload_object.unload_all,
                    log_run_time=log_run_time,
                )
                torch.cuda.empty_cache()
        self.active_dit_model_name = None

    def _encode_video_latents_for_sds(self, pred_rgb, use_tiny_vae=False):
        if use_tiny_vae:
            return self._encode_video_latents_tiny(pred_rgb)
        return self._encode_video_latents(pred_rgb)

    def _prepare_model_for_timestep(self, timestep, boundary, offload_model, log_run_time=False):
        if timestep.item() >= boundary:
            required_model_name = "high_noise_model"
            offload_model_name = "low_noise_model"
        else:
            required_model_name = "low_noise_model"
            offload_model_name = "high_noise_model"

        self.active_dit_model_name = required_model_name
        if self.enable_mmgp:
            return self._ensure_mmgp_model(required_model_name)

        if offload_model or self.init_on_cpu:
            if next(getattr(self, offload_model_name).parameters()).device.type == "cuda":
                getattr(self, offload_model_name).to("cpu")
            self._move_model_to_forward_device(
                getattr(self, required_model_name),
                log_run_time=log_run_time,
            )

        return getattr(self, required_model_name)

    def _compute_latent_shape(self, size, frame_num):
        width, height = size
        aspect_ratio = height / width
        max_area = height * width

        lat_h = round(
            np.sqrt(max_area * aspect_ratio)
            // self.vae_stride[1]
            // self.patch_size[1]
            * self.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio)
            // self.vae_stride[2]
            // self.patch_size[2]
            * self.patch_size[2]
        )
        height = lat_h * self.vae_stride[1]
        width = lat_w * self.vae_stride[2]
        max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w
        max_seq_len //= self.patch_size[1] * self.patch_size[2]
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        return width, height, lat_w, lat_h, max_seq_len

    def _encode_reference_condition(self, pred_rgb, ref_img=None):
        with torch.no_grad():
            if ref_img is None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub(0.5).div(0.5).to(self.vae_to_device)
            ref_frame = torch.nn.functional.interpolate(
                ref_img[None].detach().clone().cpu(),
                size=(self.h, self.w),
                mode="bicubic",
            ).transpose(0, 1)
            padding = torch.zeros(
                3,
                self.frame_num - 1,
                self.h,
                self.w,
                device=ref_frame.device,
                dtype=ref_frame.dtype,
            )
            y = self.vae.encode(
                [
                    torch.concat(
                        [ref_frame, padding],
                        dim=1,
                    ).to(self.device)
                ]
            )[0]
            return torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)

    def _encode_video_latents(self, pred_rgb):
        pred_rgb = pred_rgb * 2.0 - 1.0
        pred_rgb = pred_rgb.permute(1, 0, 2, 3)
        return self.vae.encode([pred_rgb])[0]

    def load_tiny_vae(self, vae_path="tiny_vae/lighttaew2_1.pth", dtype=torch.float32):
        if not os.path.exists(vae_path):
            raise FileNotFoundError(
                f"Tiny VAE checkpoint not found: {vae_path}\n"
                "Download the compatible checkpoint with:\n"
                "  huggingface-cli download lightx2v/Autoencoders lighttaew2_1.pth --local-dir ./tiny_vae/"
            )

        try:
            from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny
        except ImportError as exc:
            raise ImportError(
                "Using --use_tiny_vae requires the vendored lightx2v tiny-VAE module."
            ) from exc

        device = getattr(self, "vae_to_device", self.device)
        self.tiny_vae = WanVAE_tiny(
            vae_path=vae_path,
            dtype=dtype,
            device=device,
            need_scaled=True,
        )
        self.tiny_vae.to(device).eval().requires_grad_(False)
        if hasattr(self.tiny_vae, "prepare_latent_tensor"):
            self.tiny_vae.prepare_latent_tensor()
        elif getattr(self.tiny_vae, "need_scaled", False):
            self.tiny_vae.latents_mean_tensor = torch.tensor(self.tiny_vae.latents_mean).view(
                1, self.tiny_vae.z_dim, 1, 1, 1
            ).to(device, torch.float32)
            self.tiny_vae.latents_std_tensor = 1.0 / torch.tensor(self.tiny_vae.latents_std).view(
                1, self.tiny_vae.z_dim, 1, 1, 1
            ).to(device, torch.float32)

    def _encode_video_latents_tiny(self, pred_rgb):
        if not hasattr(self, "tiny_vae"):
            raise RuntimeError("Tiny VAE is not loaded. Call load_tiny_vae() first.")
        pred_rgb = pred_rgb.unsqueeze(0)
        if hasattr(self.tiny_vae, "encode"):
            latents = self.tiny_vae.encode(pred_rgb)
        else:
            latents = self.tiny_vae.taehv.encode_video(pred_rgb)
            if getattr(self.tiny_vae, "need_scaled", False):
                mean = getattr(self.tiny_vae, "latents_mean_tensor", None)
                std = getattr(self.tiny_vae, "latents_std_tensor", None)
                if mean is None or std is None:
                    mean = torch.tensor(self.tiny_vae.latents_mean).view(1, self.tiny_vae.z_dim, 1, 1, 1)
                    std = 1.0 / torch.tensor(self.tiny_vae.latents_std).view(1, self.tiny_vae.z_dim, 1, 1, 1)
                mean = mean.to(latents.device, latents.dtype)
                std = std.to(latents.device, latents.dtype)
                latents = latents.transpose(1, 2)
                latents = (latents - mean) * std
                latents = latents.transpose(1, 2)
        return latents.squeeze(0).permute(1, 0, 2, 3)

    def _sample_timestep(self, step_ratio):
        if step_ratio is not None:
            if not self.resample_timestep:
                timestep = (1.0 - step_ratio) * self.num_train_timesteps
                timestep = min(max(timestep, self.min_step), self.max_step)
                return torch.full((1,), timestep, dtype=torch.float32, device=self.device)

            step_idx = int((1.0 - step_ratio) * self.sds_iterations)
            step_idx = max(0, min(step_idx, len(self.resample_ts) - 1))
            return torch.full(
                (1,),
                self.resample_ts[step_idx],
                dtype=torch.float32,
                device=self.device,
            ) * self.num_train_timesteps

        if not self.resample_timestep:
            return torch.randint(
                self.min_step,
                self.max_step + 1,
                (1,),
                dtype=torch.float32,
                device=self.device,
            )

        step_idx = random.randint(0, len(self.resample_ts) - 1)
        return torch.full(
            (1,),
            self.resample_ts[step_idx],
            dtype=torch.float32,
            device=self.device,
        ) * self.num_train_timesteps

    def compute_sds_targets(
        self,
        pred_rgbs_detached,
        step_ratio=None,
        ref_imgs=None,
        guidance_scale=100.0,
        pred_flow=False,
        log_run_time=False,
        use_tiny_vae=False,
    ):
        timestep = self._sample_timestep(step_ratio)
        targets = []

        for idx, pred_rgb in enumerate(pred_rgbs_detached):
            ref_img = None if ref_imgs is None else ref_imgs[idx]

            if log_run_time:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()

            y = self._encode_reference_condition(pred_rgb, ref_img=ref_img)
            with torch.no_grad():
                target_latents = self._encode_video_latents_for_sds(
                    pred_rgb,
                    use_tiny_vae=use_tiny_vae,
                )

            if log_run_time:
                ender.record()
                torch.cuda.synchronize()
                print(f"[Timing] VAE encode: {starter.elapsed_time(ender) / 1000.0}")
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()

            with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
                noise = torch.randn_like(target_latents)
                t_ratio = timestep.item() / self.num_train_timesteps
                latents_noisy = target_latents * (1.0 - t_ratio) + t_ratio * noise

                weight = 1.0 if self.resample_timestep else _w_logit_normal(t_ratio)
                arg_c = dict(self.arg_c)
                arg_null = dict(self.arg_null)
                arg_c["y"] = [y]
                arg_null["y"] = [y]

                latents_noisy_unet = latents_noisy.detach().to(self.unet_to_device)
                timestep_unet = timestep.to(self.unet_to_device)
                model = self._prepare_model_for_timestep(
                    timestep_unet,
                    self.boundary * self.num_train_timesteps,
                    True,
                    log_run_time=log_run_time,
                )

                if log_run_time:
                    ender.record()
                    torch.cuda.synchronize()
                    print(f"[Timing] SDS prep: {starter.elapsed_time(ender) / 1000.0}")
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    starter.record()

                noise_pred_cond = model([latents_noisy_unet], t=timestep_unet, **arg_c)[0]
                noise_pred_uncond = model([latents_noisy_unet], t=timestep_unet, **arg_null)[0]

                if log_run_time:
                    ender.record()
                    torch.cuda.synchronize()
                    print(f"[Timing] DiT inference: {starter.elapsed_time(ender) / 1000.0}")

                noise_pred_cond = noise_pred_cond.detach().to(target_latents.device)
                noise_pred_uncond = noise_pred_uncond.detach().to(target_latents.device)

                if pred_flow:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise_target = noise - target_latents
                else:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise_pred = latents_noisy + (1.0 - t_ratio) * flow
                    noise_target = noise

            grad = weight * (noise_pred - noise_target)
            grad = torch.nan_to_num(grad)
            targets.append((target_latents - grad).detach())

        return targets

    def sds_loss_from_target(self, pred_rgb, target, use_tiny_vae=False):
        latents = self._encode_video_latents_for_sds(
            pred_rgb,
            use_tiny_vae=use_tiny_vae,
        )
        return 0.5 * F.mse_loss(latents.float(), target.to(latents.device).float(), reduction="sum")

    def sds_setup(
        self,
        input_prompt,
        t_range=(0.02, 0.98),
        frame_num=81,
        size=(1280, 720),
        unet_to_device="cuda",
        vae_to_device="cuda",
        resample_timestep=False,
        sds_iterations=3000,
        n_prompt="",
        del_decoder=False,
    ):
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.frame_num = frame_num
        self.unet_to_device = unet_to_device
        self.vae_to_device = vae_to_device
        self.resample_timestep = resample_timestep
        self.sds_iterations = sds_iterations

        self.w, self.h, self.lat_w, self.lat_h, max_seq_len = self._compute_latent_shape(
            size=size,
            frame_num=frame_num,
        )

        if not n_prompt or n_prompt == ".":
            n_prompt = self.sample_neg_prompt

        if not self.t5_cpu:
            self.text_encoder.model.to(unet_to_device)
            context = self.text_encoder([input_prompt], unet_to_device)
            context_null = self.text_encoder([n_prompt], unet_to_device)
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [tensor.to(unet_to_device) for tensor in context]
            context_null = [tensor.to(unet_to_device) for tensor in context_null]

        self.arg_c = {"context": [context[0]], "seq_len": max_seq_len}
        self.arg_null = {"context": context_null, "seq_len": max_seq_len}

        if self.resample_timestep:
            sampled_t = _sample_weighted_logit_normal_timesteps(
                mu=0.0,
                sigma=1.0,
                total_steps=sds_iterations + 1,
            )
            self.resample_ts = np.clip(sampled_t, t_range[0], t_range[1])
        else:
            self.resample_ts = None

        init_mask = torch.ones(1, frame_num, self.lat_h, self.lat_w, device=self.device)
        init_mask[:, 1:] = 0
        init_mask = torch.concat(
            [
                torch.repeat_interleave(init_mask[:, 0:1], repeats=4, dim=1),
                init_mask[:, 1:],
            ],
            dim=1,
        )
        init_mask = init_mask.view(1, init_mask.shape[1] // 4, 4, self.lat_h, self.lat_w)
        self.init_mask = init_mask.transpose(1, 2)[0].to(self.unet_to_device)

        self.text_encoder = None
        if del_decoder:
            self.vae.model.decoder.to("cpu")

    def sds_step(
        self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pred_flow=False,
        log_run_time=False,
        use_tiny_vae=False,
    ):
        if log_run_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()

        y = self._encode_reference_condition(pred_rgb, ref_img=ref_img)
        latents = self._encode_video_latents_for_sds(pred_rgb, use_tiny_vae=use_tiny_vae)
        timestep = self._sample_timestep(step_ratio)

        if log_run_time:
            ender.record()
            torch.cuda.synchronize()
            print(f"[Timing] VAE encode: {starter.elapsed_time(ender) / 1000.0}")
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()

        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            noise = torch.randn_like(latents)
            t_ratio = timestep.item() / self.num_train_timesteps
            latents_noisy = latents * (1.0 - t_ratio) + t_ratio * noise

            weight = 1.0 if self.resample_timestep else _w_logit_normal(t_ratio)
            arg_c = dict(self.arg_c)
            arg_null = dict(self.arg_null)
            arg_c["y"] = [y]
            arg_null["y"] = [y]

            latents_noisy_unet = latents_noisy.detach().to(self.unet_to_device)
            timestep_unet = timestep.to(self.unet_to_device)
            model = self._prepare_model_for_timestep(
                timestep_unet,
                self.boundary * self.num_train_timesteps,
                True,
                log_run_time=log_run_time,
            )

            if log_run_time:
                ender.record()
                torch.cuda.synchronize()
                print(f"[Timing] SDS prep: {starter.elapsed_time(ender) / 1000.0}")
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()

            noise_pred_cond = model([latents_noisy_unet], t=timestep_unet, **arg_c)[0]
            noise_pred_uncond = model([latents_noisy_unet], t=timestep_unet, **arg_null)[0]

            if log_run_time:
                ender.record()
                torch.cuda.synchronize()
                print(f"[Timing] DiT inference: {starter.elapsed_time(ender) / 1000.0}")

            noise_pred_cond = noise_pred_cond.detach().to(latents.device)
            noise_pred_uncond = noise_pred_uncond.detach().to(latents.device)

            if pred_flow:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_target = noise - latents
            else:
                flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_pred = latents_noisy + (1.0 - t_ratio) * flow
                noise_target = noise

        if self.enable_mmgp:
            self.release_active_dit_after_target(log_run_time=log_run_time)

        grad = weight * (noise_pred - noise_target)
        grad = torch.nan_to_num(grad)
        target = (latents - grad).detach()

        return 0.5 * F.mse_loss(latents.float(), target, reduction="sum")
