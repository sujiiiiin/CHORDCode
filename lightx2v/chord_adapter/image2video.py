# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from wan.distributed.fsdp import shard_model
from wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from wan.distributed.util import get_world_size
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    get_sampling_sigmas_rev,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import torch.nn.functional as F
from scipy.stats import norm
import scipy.stats
from random import randint
from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny
from .wan22_moe_i2v_distill import LightX2VWan22MoeI2VDistill

def logit_normal_pdf(t, mu=0.0, sigma=1.0):
    eps = 1e-6
    t = np.clip(t, eps, 1 - eps)
    logit_t = np.log(t / (1 - t))
    base_pdf = norm.pdf(logit_t, loc=mu, scale=sigma)
    return base_pdf / (t * (1 - t))

def sample_from_weighted_logit_normal(mu=0.0, sigma=1.0, T=50, N_grid=1000000):
    t_grid = np.linspace(1e-6, 1 - 1e-6, N_grid)
    x_t = logit_normal_pdf(t_grid, mu, sigma)
    w = (t_grid / (1 - t_grid)) * x_t
    w /= np.trapz(w, t_grid)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    target_cdf_vals = np.linspace(1 / T, (T - 1) / T, T - 1)
    sampled_ts = np.interp(target_cdf_vals, cdf, t_grid)
    return t_grid, w, sampled_ts

def logit(t):
    return np.log(t / (1 - t))

def pi_logit_normal(t, m, s):
    """Logit-normal density π(t; m, s)"""
    logit_t = logit(t)
    coeff = 1 / (s * np.sqrt(2 * np.pi) * t * (1 - t))
    exponent = -((logit_t - m) ** 2) / (2 * s ** 2)
    return coeff * np.exp(exponent)

def w_logit_normal(t, m=0.0, s=1.0):
    """Loss weight w(t) for Logit-Normal Sampling"""
    pi_t = pi_logit_normal(t, m, s)
    return (t / (1 - t)) * pi_t

def stratified_logit_normal_samples(mu=0.0, sigma=1.0, T=100):
    # i/T for i in 1..T-1
    cdf_vals = torch.linspace(1 / T, (T) / T, T)
    
    # Inverse normal CDF
    norm_ppf = torch.tensor(scipy.stats.norm.ppf(cdf_vals.numpy(), loc=mu, scale=sigma))
    
    # Apply sigmoid to get logit-normal samples
    samples = torch.sigmoid(norm_ppf)
    return samples

def sample_from_weighted_logit_normal_ratio(mu=0.0, sigma=1.0, T=50, ratio=1.0, N_grid=1000000):
    scaled_T = int(T /ratio)
    _, _, sampled_t = sample_from_weighted_logit_normal(
        mu=0.0, sigma=1.0, T=scaled_T + 1, N_grid=1000000
    )
    sampled_t = sampled_t[::-1]
    sampled_t = sampled_t[:T]
    sampled_t = sampled_t[::-1]
    return sampled_t

class _LightX2VTextEncoderAdapter:
    def __init__(self, t5_encoder):
        self._t5_encoder = t5_encoder
        self.model = t5_encoder.model

    def __call__(self, texts, device):
        return self._t5_encoder.infer(texts)


class _LightX2VVAEAdapter:
    def __init__(self, vae):
        self._vae = vae
        self.model = vae.model

    def encode(self, videos):
        outputs = []
        for video in videos:
            if video.dim() == 4:
                video = video.unsqueeze(0)
            outputs.append(self._vae.encode(video))
        return outputs

    def encode_nockpt(self, videos):
        return self.encode(videos)

    def decode(self, zs):
        outputs = []
        for z in zs:
            decoded = self._vae.decode(z)
            if isinstance(decoded, (list, tuple)):
                decoded = decoded[0]
            if decoded.dim() == 5 and decoded.shape[0] == 1:
                decoded = decoded.squeeze(0)
            outputs.append(decoded)
        return outputs


class _LightX2VDiTAdapter:
    def __init__(self, model, scheduler, config):
        self.model = model
        self.scheduler = scheduler
        self.config = config
        self.text_len = config["text_len"]
        self.cpu_offload = config.get("cpu_offload", False)
        self.offload_granularity = config.get("offload_granularity", "block")
        self._last_latent_shape = None
        self.model.set_scheduler(scheduler)

    def _pack_context(self, context_list):
        if context_list is None:
            raise ValueError("context is required for DiT inference.")
        if torch.is_tensor(context_list):
            context_list = [context_list]
        padded = [
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
            )
            for u in context_list
        ]
        return torch.stack(padded)

    def _ensure_cos_sin(self, latents):
        latent_shape = tuple(latents.shape)
        if self._last_latent_shape == latent_shape and getattr(self.scheduler, "cos_sin", None) is not None:
            return
        patch_size = getattr(self.scheduler, "patch_size", (1, 2, 2))
        grid_sizes = (
            latents.shape[1] // patch_size[0],
            latents.shape[2] // patch_size[1],
            latents.shape[3] // patch_size[2],
        )
        self.scheduler.cos_sin = self.scheduler.prepare_cos_sin(grid_sizes)
        self._last_latent_shape = latent_shape

    def __call__(self, latent_model_input, t=None, context=None, seq_len=None, y=None):
        if t is None:
            raise ValueError("t is required for DiT inference.")
        latents = latent_model_input[0]
        target_dtype = getattr(self.model.transformer_infer, "infer_dtype", latents.dtype)
        latents = latents.to(dtype=target_dtype)
        self.scheduler.latents = latents
        self.scheduler.timestep_input = t
        self._ensure_cos_sin(latents)

        context_tensor = self._pack_context(context).to(latents.device, dtype=target_dtype)
        y_tensor = y[0] if isinstance(y, (list, tuple)) else y
        if y_tensor is not None:
            y_tensor = y_tensor.to(latents.device, dtype=target_dtype)
        inputs = {
            "text_encoder_output": {
                "context": context_tensor,
                "context_null": context_tensor,
            },
            "image_encoder_output": {
                "clip_encoder_out": None,
                "vae_encoder_out": y_tensor,
            },
        }

        if self.cpu_offload and self.offload_granularity != "model":
            self.model.pre_weight.to_cuda()
            self.model.transformer_weights.non_block_weights_to_cuda()

        noise_pred = self.model._infer_cond_uncond(inputs, infer_condition=True)

        if self.cpu_offload and self.offload_granularity != "model":
            self.model.pre_weight.to_cpu()
            self.model.transformer_weights.non_block_weights_to_cpu()

        return [noise_pred]

    def to_cpu(self):
        self.model.to_cpu()

    def to_cuda(self):
        self.model.to_cuda()

    def to(self, device):
        device = torch.device(device)
        if device.type == "cuda":
            self.to_cuda()
        else:
            self.to_cpu()
        return self

    def cpu(self):
        self.to_cpu()

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
        off_load=True,
        is_attn2=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = torch.bfloat16

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        config_json = getattr(config, "lightx2v_config_json", None)
        self.lx_pipeline = LightX2VWan22MoeI2VDistill(
            model_path=checkpoint_dir,
            config_json=config_json,
            device_id=device_id,
            seed=0,
            is_attn2=is_attn2,
        )
        lx_config = self.lx_pipeline.config
        self.t5_cpu = self.t5_cpu or lx_config.get("t5_cpu_offload", False)
        self.lx_cpu_offload = lx_config.get("cpu_offload", False)
        self.lx_offload_granularity = lx_config.get("offload_granularity", "block")
        self.lx_boundary_step_index = lx_config.get("boundary_step_index", 0)
        self._lx_cur_model_index = None

        self.text_encoder = _LightX2VTextEncoderAdapter(self.lx_pipeline.text_encoder)
        self.vae_stride = lx_config["vae_stride"]
        self.patch_size = lx_config.get("patch_size", (1, 2, 2))
        self.vae = _LightX2VVAEAdapter(self.lx_pipeline.vae_encoder)

        logging.info(f"Creating LightX2V Wan2.2 distill models from {checkpoint_dir}")
        self.high_noise_model = _LightX2VDiTAdapter(
            self.lx_pipeline.model.model[0],
            self.lx_pipeline.scheduler,
            lx_config,
        )
        self.low_noise_model = _LightX2VDiTAdapter(
            self.lx_pipeline.model.model[1],
            self.lx_pipeline.scheduler,
            lx_config,
        )
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt
        self.off_load = off_load

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model
    
    def load_tiny_vae(self, vae_path = "./models/vae/lighttaew2_1.pth", dtype=torch.float32):
        self.tiny_vae = WanVAE_tiny(vae_path=vae_path, dtype=dtype, device="cuda", need_scaled=True)
        self.tiny_vae.to("cuda")
        self.tiny_vae.prepare_latent_tensor()

    def _prepare_model_for_timestep(self, t, boundary, offload_model, step_index=None):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if step_index is None:
            if t.item() >= boundary:
                required_model_name = 'high_noise_model'
                offload_model_name = 'low_noise_model'
            else:
                required_model_name = 'low_noise_model'
                offload_model_name = 'high_noise_model'
            required_model_index = 0 if required_model_name == 'high_noise_model' else 1
        else:
            if step_index < self.lx_boundary_step_index:
                required_model_name = 'high_noise_model'
                offload_model_name = 'low_noise_model'
                required_model_index = 0
            else:
                required_model_name = 'low_noise_model'
                offload_model_name = 'high_noise_model'
                required_model_index = 1
        if self.lx_cpu_offload and self.lx_offload_granularity == "model":
            if offload_model or self.init_on_cpu:
                if self._lx_cur_model_index is None:
                    getattr(self, required_model_name).to_cuda()
                elif self._lx_cur_model_index != required_model_index:
                    getattr(self, offload_model_name).to_cpu()
                    getattr(self, required_model_name).to_cuda()
                self._lx_cur_model_index = required_model_index
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        msk = msk.to(self.param_dtype)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = y.to(self.param_dtype)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            y = y.to(self.param_dtype)
            

            arg_c = {
                'context': [context[0].to(self.param_dtype)],
                'seq_len': max_seq_len,
                'y': [y],
            }

            # arg_null = {
            #     'context': context_null,
            #     'seq_len': max_seq_len,
            #     'y': [y],
            # }

            if offload_model:
                torch.cuda.empty_cache()

            for step_index, t in enumerate(tqdm(timesteps, disable=True)):
                latent_model_input = [latent.to(self.device, dtype=self.param_dtype)]
                timestep = [t.to(self.device, dtype=self.param_dtype)]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]
                # print("print timestep shape", timestep.shape)
                
                noise_pred = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                
                

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep


            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def generate_with_tensor(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = img.sub_(0.5).div_(0.5)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        msk = msk.to(self.param_dtype)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = y.to(self.param_dtype)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            y = y.to(self.param_dtype)
            

            arg_c = {
                'context': [context[0].to(self.param_dtype)],
                'seq_len': max_seq_len,
                'y': [y],
            }

            # arg_null = {
            #     'context': context_null,
            #     'seq_len': max_seq_len,
            #     'y': [y],
            # }

            if offload_model:
                torch.cuda.empty_cache()

            for step_index, t in enumerate(tqdm(timesteps, disable=True)):
                latent_model_input = [latent.to(self.device, dtype=self.param_dtype)]
                timestep = [t.to(self.device, dtype=self.param_dtype)]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]
                # print("print timestep shape", timestep.shape)
                
                noise_pred = model(
                    latent_model_input, t=timestep, **arg_c)[0]
                
                

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep


            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
    
    def sds_setup(self, input_prompt, t_range=[0.02, 0.98], sample_solver='unipc', frame_num=81, size=(1280, 720)
                  , unet_to_device="cuda", vae_to_device="cuda", resample_timestep = False, sds_iterations=3000, n_prompt="",
                  del_decoder=False, store_noise=False, sample_view_num = 1, use_shift_flow=False, step_ratio=1.0):
        seed = random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        # scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        
        F = frame_num
        w = size[0]
        h = size[1]
        aspect_ratio = h / w
        max_area = h * w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        self.h = h
        self.w = w
        self.lat_w = lat_w
        self.lat_h = lat_h
        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        
        self.unet_to_device = unet_to_device
        
        n_prompt = self.sample_neg_prompt
        
        if n_prompt == "" or n_prompt == ".":
            n_prompt = self.sample_neg_prompt
        print("print input_prompt: ", input_prompt)
        print("print n_prompt: ", n_prompt)
        if not self.t5_cpu:
            self.text_encoder.model.to(unet_to_device)
            context = self.text_encoder([input_prompt], unet_to_device)
            context_null = self.text_encoder([n_prompt], unet_to_device)
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            
        context = [t.to(unet_to_device) for t in context]
        context_null = [t.to(unet_to_device) for t in context_null]

        self.arg_c = {'context': [context[0]], 'seq_len': max_seq_len}
        self.arg_null = {'context': context_null, 'seq_len': max_seq_len}
        
        self.resample_timestep = resample_timestep
        if resample_timestep:
            if use_shift_flow == False:
                self.sds_iterations = sds_iterations
                if step_ratio == 1.0:
                    _, _, sampled_t = sample_from_weighted_logit_normal(mu=0.0, sigma=1.0, T=sds_iterations+1, N_grid=1000000)
                else:
                    sampled_t = sample_from_weighted_logit_normal_ratio(
                            mu=0.0, sigma=1.0, T=sds_iterations, ratio=step_ratio, N_grid=1000000
                        )
                sampled_t = np.clip(sampled_t, t_range[0], t_range[1])
                self.resample_ts = sampled_t
            else:
                self.sds_iterations = sds_iterations
                sampled_t = get_sampling_sigmas_rev(self.sds_iterations+1, 3.0)
                sampled_t = np.clip(sampled_t, t_range[0], t_range[1])
                self.resample_ts = sampled_t
        
        
        self.frame_num = frame_num
        
        
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0].to(self.unet_to_device)
        msk = msk.to(self.param_dtype)
        
        self.init_mask = msk
        self.vae_to_device=vae_to_device
        del self.text_encoder
        if del_decoder:
            self.vae.model.decoder.to("cpu")
        
        self.store_noise = False
        if store_noise:
            self.store_noise = True
            self.noise_list = []
            for _ in range(sample_view_num):
                cur_noise = torch.randn((16, (F - 1) // self.vae_stride[0] + 1, lat_h, lat_w), dtype=torch.float32, device="cuda")
                self.noise_list.append(cur_noise)

    def set_hw(self, h, w):
        aspect_ratio = h / w
        max_area = h * w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        self.h = h
        self.w = w
        
    def test_tiny_vae(self, video):
        latent_video = self.tiny_vae.encode(video.unsqueeze(0))
        latent_video = latent_video[0].permute(1,0,2,3)
        print("print latent_video shape: ", latent_video.shape)
        with torch.no_grad():
            raw_video = self.vae.decode([latent_video])[0]
        raw_video = (raw_video + 1.0) * 0.5
        raw_video = raw_video.permute(1,0,2,3)
        raw_video = raw_video.clamp(0,1).detach().clone()
        return raw_video, latent_video
    
    def offload_model(self):
        self.low_noise_model.cpu()
        self.high_noise_model.cpu()
    def onload_model(self):
        return
    
    def get_z(self, ref_img):
        with torch.no_grad():
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
            z = self.vae.encode([ref_img])
            return z[0].detach().clone()
        
    def encode_video(self, pred_rgb):
        with torch.no_grad():
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb])[0]
            return latents.detach()
        
    def calc_sds_loss(self, pred_rgb, target, use_tiny_vae=False):
        if use_tiny_vae:
            latents = self.tiny_vae.encode(pred_rgb.unsqueeze(0))
            latents = latents.squeeze(0).permute(1,0,2,3)
        else:
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb])[0]
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        return loss
    
    def calc_sds_loss_batch(self, pred_rgb, target):
        # print("print pred_rgb shape", pred_rgb.shape)
        latents = self.tiny_vae.encode(pred_rgb)        
        # print("print latents shape: ", latents.shape)
        # print("print target shape: ", target.shape)
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        return loss
    
    def get_latents(self, pred_rgb):
        pred_rgb = pred_rgb * 2.0 - 1.0
        pred_rgb = pred_rgb.permute(1,0,2,3)
        # pred_rgb = pred_rgb[0:3]
        latents = self.vae.encode([pred_rgb])[0]
        return latents.detach()

    def sds_step(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False,
        view_id = -1,
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            if self.store_noise and view_id >= 0:
                noise = self.noise_list[view_id]
            else:
                noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
                
            noise_pred_uncond = model(
                [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            if save_mid:
                if use_edit is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                pred_original = latents_noisy - t_ratio * flow
                self.saved_pred_original = pred_original.detach().clone()
                # self.saved_latents = latents.detach().clone()
            if use_edit is False:
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
            else:
                noise_pred = noise_pred_cond
                noise = noise_pred_uncond
                w = guidance_scale
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        
        return loss
    
    def sds_step_tinyvae(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False,
        view_id = -1,
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            latents = self.tiny_vae.encode(pred_rgb.unsqueeze(0))
            latents = latents.squeeze(0).permute(1,0,2,3)
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            if self.store_noise and view_id >= 0:
                noise = self.noise_list[view_id]
            else:
                noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
                
            noise_pred_uncond = model(
                [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            if save_mid:
                if use_edit is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                pred_original = latents_noisy - t_ratio * flow
                self.saved_pred_original = pred_original.detach().clone()
                # self.saved_latents = latents.detach().clone()
            if use_edit is False:
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
            else:
                noise_pred = noise_pred_cond
                noise = noise_pred_uncond
                w = guidance_scale
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        
        return loss
    
    def sds_step_with_target(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False,
        use_tiny_vae = False,
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            if use_tiny_vae:
                latents = self.tiny_vae.encode(pred_rgb.unsqueeze(0))
                latents = latents.squeeze(0).permute(1,0,2,3)
            else:
                pred_rgb = pred_rgb * 2.0 - 1.0
                pred_rgb = pred_rgb.permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                
                print("print w : ", w)
            # print("print t_ratio : ", t_ratio)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
                
            # noise_pred_uncond = model(
            #     [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            # noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            if save_mid:
                flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                pred_original = latents_noisy - t_ratio * flow
                self.saved_pred_original = pred_original.detach().clone()
                # self.saved_latents = latents.detach().clone()
            if pred_flow is False:
                flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                noise_pred = latents_noisy + (1 - t_ratio) * flow
            else:
                noise_pred = noise_pred_cond
                noise = noise - latents
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        
        return loss, target
    
    def sds_step_with_target_multi_step(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False,
        use_tiny_vae = False,
        steps = 1,
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            if use_tiny_vae:
                latents = self.tiny_vae.encode(pred_rgb.unsqueeze(0))
                latents = latents.squeeze(0).permute(1,0,2,3)
            else:
                pred_rgb = pred_rgb * 2.0 - 1.0
                pred_rgb = pred_rgb.permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                
                print("print w : ", w)
            # print("print t_ratio : ", t_ratio)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                
            timesteps = []
            sample_shift = 3.0
            for i in range(steps+1):
                step_t = i / steps
                step_t = np.clip(step_t, 0.0, 1.0)
                step_t = sample_shift * step_t / (1 + (sample_shift - 1) * step_t)
                step_t = step_t * t_ratio * self.num_train_timesteps
                timesteps.append(step_t)
                
            # reverse order
            timesteps = timesteps[::-1]
            for i in range(steps):
                cur_step = timesteps[i]
                next_step = timesteps[i+1]
                cur_t = cur_step / self.num_train_timesteps
                next_t = next_step / self.num_train_timesteps
                cur_step_tensor = torch.full((batch_size,), cur_step, dtype=torch.float32, device="cuda")
                model = self._prepare_model_for_timestep(
                    cur_step_tensor, self.boundary * self.num_train_timesteps, True)
                # print("print cur_step: ", cur_step)
                flow = model(
                        [latents_noisy], t=cur_step_tensor, **arg_c)[0]
                latents_noisy = latents_noisy + (next_t - cur_t) * flow
                
            # noise_pred_uncond = model(
            #     [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)

        target = latents_noisy.detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')
        
        return loss, target
    
    def sds_step_only_target(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            with torch.no_grad():
                pred_rgb = pred_rgb * 2.0 - 1.0
                pred_rgb = pred_rgb.permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode_nockpt([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
                
            noise_pred_uncond = model(
                [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            if save_mid:
                if use_edit is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                pred_original = latents_noisy - t_ratio * flow
                self.saved_pred_original = pred_original.detach().clone()
                # self.saved_latents = latents.detach().clone()
            if use_edit is False:
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
            else:
                noise_pred = noise_pred_cond
                noise = noise_pred_uncond
                w = guidance_scale
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        return target
    
    def sds_step_only_target_with_latent(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            with torch.no_grad():
                pred_rgb = pred_rgb * 2.0 - 1.0
                pred_rgb = pred_rgb.permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode_nockpt([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
                
            noise_pred_uncond = model(
                [latents_noisy], t=t, **arg_null)[0]
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            if save_mid:
                if use_edit is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    flow = noise_pred_cond
                latents_noisy = latents_noisy.detach().to("cuda")
                pred_original = latents_noisy - t_ratio * flow
                self.saved_pred_original = pred_original.detach().clone()
                # self.saved_latents = latents.detach().clone()
            if use_edit is False:
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
            else:
                noise_pred = noise_pred_cond
                noise = noise_pred_uncond
                w = guidance_scale
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        return target, latents
    
    def sds_step_batch(self,
        pred_rgb_list,
        step_ratio=None,
        ref_img_list=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        
        pred_rgb_list = [x.detach() for x in pred_rgb_list]
        
        with torch.no_grad():
            y_list = []
            for i in range(len(pred_rgb_list)):
                pred_rgb = pred_rgb_list[i]
                ref_img = None
                if ref_img_list is not None:
                    ref_img = ref_img_list[i]
                else:
                    ref_img = pred_rgb[0].detach().clone()
                ref_img = ref_img.sub_(0.5).div_(0.5)
                y = self.vae.encode_nockpt([
                    torch.concat([
                            torch.nn.functional.interpolate(
                                ref_img[None], size=(self.h, self.w), mode='bicubic').transpose(
                                    0, 1),
                            torch.zeros(3, self.frame_num-1, self.h, self.w, device="cuda")
                        ],
                                    dim=1)
                    ])[0]
                y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
                y_list.append(y)
                    
            # if ref_img==None:
            #     ref_img = pred_rgb[0].detach().clone()
            # ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            # y = self.vae.encode([
            # torch.concat([
            #         torch.nn.functional.interpolate(
            #             ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
            #                 0, 1),
            #         torch.zeros(3, self.frame_num-1, self.h, self.w)
            #     ],
            #                 dim=1).to("cuda")
            # ])[0]
            # y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            with torch.no_grad():
                for i in range(len(pred_rgb_list)):
                    pred_rgb_list[i] = pred_rgb_list[i] * 2.0 - 1.0
                    pred_rgb_list[i] = pred_rgb_list[i].permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents_list = self.vae.encode_nockpt(pred_rgb_list)
        else:
            latents_list = pred_rgb
        
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
        batch_size = len(pred_rgb_list)
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise_list = [torch.randn_like(latents_list[i]) for i in range(batch_size)]
            t_ratio = t[0].item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy_list = []
            for i in range(batch_size):
                cur_latents_noisy = latents_list[i]*(1.0-t_ratio) + t_ratio*noise_list[i]
                cur_latents_noisy = cur_latents_noisy.detach().to(self.unet_to_device)
                latents_noisy_list.append(cur_latents_noisy)
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            for _ in range(batch_size-1):
                arg_c['context'].append(arg_c['context'][0].clone())
                arg_null['context'].append(arg_null['context'][0].clone())
            arg_c['y'] = y_list
            arg_null['y'] = y_list
            
            # t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t[0:1], self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond_list = model(
                        latents_noisy_list, t=t, **arg_c)
                
            noise_pred_uncond_list = model(
                latents_noisy_list, t=t, **arg_null)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            target_list = []
            for i in range(batch_size):
            
                noise_pred_cond = noise_pred_cond_list[i].detach().to("cuda")
                noise_pred_uncond = noise_pred_uncond_list[i].detach().to("cuda")
                noise = noise_list[i].to("cuda")
                latents = latents_list[i].to("cuda")
                latents_noisy = latents_noisy_list[i].to("cuda")
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
                    
                grad = w * (noise_pred - noise)
                grad = torch.nan_to_num(grad)
                target = (latents - grad).detach()
                target_list.append(target)
                
            return target_list
        
    def decode_vae_latents(self, latents, show_time=False):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        with torch.no_grad():
            raw_video = self.vae.decode([latents])[0]
            raw_video = (raw_video + 1.0) * 0.5
            raw_video = raw_video.permute(1,0,2,3)
            raw_video = raw_video.clamp(0,1).detach().clone()
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae decode time: ", ms / 1000.0)
        return raw_video
            
            
    
    def i2v_save_feat(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 timestep_num=1,):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        msk = msk.to(self.param_dtype)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = y.to(self.param_dtype)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        cnt = 0
        
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            for step_index, t in enumerate(tqdm(timesteps, disable=True)):
                latent_model_input = [latent.to(self.device, dtype=self.param_dtype)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model, step_index=step_index)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]
                
                cnt = cnt + 1
                if cnt == timestep_num:
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, save_mid=True, **arg_c)[0]
                    print("print final timestep: ", t)
                    saved_mid = model.get_saved_mid()
                    return saved_mid
                else:
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **arg_c)[0]

                # noise_pred_cond = model(
                #     latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
            
    def i2v_save_keys(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                #  timestep_num=1,
                 save_timesteps=[],
                 save_block_idx=[],
                 save_keys=[]):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            (F - 1) // self.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        msk = msk.to(self.param_dtype)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = y.to(self.param_dtype)
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        cnt = 0
        
        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()
                
            saved_feats = {}

            for step_index, t in enumerate(tqdm(timesteps, disable=True)):
                latent_model_input = [latent.to(self.device, dtype=self.param_dtype)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model, step_index=step_index)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]
                
                cnt = cnt + 1
                if cnt in save_timesteps:
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, save_mid=True, save_block_idx=save_block_idx, save_keys=save_keys, **arg_c)[0]
                    # print("print final timestep: ", t)
                    saved_mid = model.get_saved_mid()
                    saved_feats[cnt] = saved_mid
                else:
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **arg_c)[0]

                # noise_pred_cond = model(
                #     latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        return videos, saved_feats
            
    def motion_sds_step(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        use_edit = False,
        sds_w = 1.0,
        multistep_sample = False,
        show_time = False,
        extract_block_idxs = []
        ):
        if show_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
        pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device)
            y = self.vae.encode([
            torch.concat([
                    torch.nn.functional.interpolate(
                        ref_img[None].detach().clone().cpu(), size=(self.h, self.w), mode='bicubic').transpose(
                            0, 1),
                    torch.zeros(3, self.frame_num-1, self.h, self.w)
                ],
                            dim=1).to("cuda")
            ])[0]
            y = torch.concat([self.init_mask, y]).detach().to(self.unet_to_device)
            
        
        if pass_encoder:
            with torch.no_grad():
                pred_rgb = pred_rgb * 2.0 - 1.0
                pred_rgb = pred_rgb.permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode_nockpt([pred_rgb])[0]
        else:
            latents = pred_rgb
            
        if show_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print("print vae encode time: ", ms / 1000.0)
            
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            
            
        batch_size = 1
        if step_ratio is not None:
            if self.resample_timestep is False:
                t = ((1 - step_ratio) * self.num_train_timesteps)
                t = min(max(t, self.min_step), self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
            else:
                t = int((1 - step_ratio) * self.sds_iterations)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps
        else:
            if self.resample_timestep is False:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.float32, device=self.device)
            else:
                t = randint(0, self.resample_ts.shape[0]-1)
                t = torch.full((batch_size,), self.resample_ts[t], dtype=torch.float32, device=self.device) * self.num_train_timesteps

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            t_ratio = t.item() / self.num_train_timesteps
            if save_mid:
                print("print t_ratio : ", t_ratio)
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                print("print t_ratio : ", t_ratio)
                print("print w : ", w)
            self.saved_w = w
            # get args
            arg_c = self.arg_c
            arg_null = self.arg_null
            arg_c['y'] = [y]
            arg_null['y'] = [y]
            arg_c['save_mid'] = True
            arg_null['save_mid'] = True
            arg_c['extract_block_idxs'] = extract_block_idxs
            arg_null['extract_block_idxs'] = extract_block_idxs
            # arg_c['save_keys'] = ["saved_y"]
            
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            model = self._prepare_model_for_timestep(
                    t, self.boundary * self.num_train_timesteps, True)
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print prepare sds time: ", ms / 1000.0)
                
                starter = torch.cuda.Event(enable_timing=True)
                ender   = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
            
            noise_pred_cond = model(
                        [latents_noisy], t=t, **arg_c)[0]
            
                
            # noise_pred_uncond = model(
            #     [latents_noisy], t=t, **arg_null)[0]
            # self.saved_mid_uncond = model.get_saved_mid()
            
            if show_time:
                ender.record()
                torch.cuda.synchronize()
                ms = starter.elapsed_time(ender)
                print("print infernece DiT time: ", ms / 1000.0)
            
            return model.get_saved_mid()[extract_block_idxs[0]]
            
            # noise_pred_cond = noise_pred_cond.detach().to("cuda")
            # noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            

            # if save_mid:
            #     if use_edit is False:
            #         flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            #     else:
            #         flow = noise_pred_cond
            #     latents_noisy = latents_noisy.detach().to("cuda")
            #     pred_original = latents_noisy - t_ratio * flow
            #     self.saved_pred_original = pred_original.detach().clone()
            #     # self.saved_latents = latents.detach().clone()
            # if use_edit is False:
            #     if pred_flow is False:
            #         flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            #         latents_noisy = latents_noisy.detach().to("cuda")
            #         noise_pred = latents_noisy + (1 - t_ratio) * flow
            #     else:
            #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            #         noise = noise - latents
            # else:
            #     noise_pred = noise_pred_cond
            #     noise = noise_pred_uncond
            #     w = guidance_scale
                
        

        # grad = w * (noise_pred - noise)
        # grad = torch.nan_to_num(grad)
        
        
        # if save_mid:
        #     self.saved_grad = grad.detach().clone()

        # target = (latents - grad).detach()
        
        # return
    
