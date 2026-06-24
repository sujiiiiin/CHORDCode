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

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like
import torch.nn.functional as F
from scipy.stats import norm
import scipy.stats
from random import randint
import numpy as np
from wan.utils.utils import cache_video, str2bool

def logit_normal_pdf(t, mu=0.0, sigma=1.0):
    eps = 1e-6
    t = np.clip(t, eps, 1 - eps)
    logit_t = np.log(t / (1 - t))
    base_pdf = norm.pdf(logit_t, loc=mu, scale=sigma)
    return base_pdf / (t * (1 - t))

def sample_from_weighted_logit_normal(mu=0.0, sigma=1.0, T=50, N_grid=1000000, weight=1.0):
    t_grid = np.linspace(1e-6, 1 - 1e-6, N_grid)
    x_t = logit_normal_pdf(t_grid, mu, sigma)
    w = (t_grid / (1 - t_grid)) * x_t
    w /= np.trapz(w, t_grid)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    target_cdf_vals = np.linspace(1.0 - ((T - 1) / T)*weight, 1.0 - (1 / T)*weight, T - 1)
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

class WanTI2V:

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
        off_load=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

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
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device, dtype=torch.float32)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.off_load = off_load
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

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

    def _move_model_to_forward_device(self):
        if hasattr(self.model, "prepare_dit_for_forward"):
            self.model.prepare_dit_for_forward(self.unet_to_device)
        else:
            self.model.to(self.unet_to_device)

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

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

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

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
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
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
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "" or n_prompt == ".":
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

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

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
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # flow = noise_pred
                # predicted_latent = latent - t/self.num_train_timesteps * flow
                # predicted_latent = (1. - mask2[0]) * z[0] + mask2[0] * predicted_latent
                # save_path = "temp_latent"
                # self.model.cpu()
                # videos = self.vae.decode([predicted_latent])
                # os.makedirs(save_path, exist_ok=True)
                # cache_video(
                #     tensor=videos[0][None],
                #     save_file=os.path.join(save_path, f"video_{t}.mp4"),
                #     fps=16,
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1))
                # self.model.to(self.device)
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
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

        return videos[0] if self.rank == 0 else None
    
    
    
    

    def sds_setup(self, input_prompt, t_range=[0.02, 0.98], sample_solver='unipc', frame_num=81, size=(1280, 720)
                  , unet_to_device="cuda", vae_to_device="cuda", resample_timestep = False, sds_iterations=3000, n_prompt="",
                  del_decoder=False):
        seed = random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        F = frame_num
        w = size[0]
        h = size[1]
        ih, iw = h, w
        
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        max_area = 704 * 1280
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        
        self.w = round(iw * scale)
        self.h = round(ih * scale)
        
        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size
        
        self.seq_len = seq_len
                    
        self.lat_h = oh // self.vae_stride[1]
        self.lat_w = ow // self.vae_stride[2]
        
        # if sample_solver == 'unipc':
        #     scheduler = FlowUniPCMultistepScheduler(
        #         num_train_timesteps=self.num_train_timesteps,
        #         shift=1,
        #         use_dynamic_shifting=False)
        #     # scheduler.set_timesteps(
        #     #         self.num_train_timesteps, device=self.device, shift=flow_shift)
        # elif sample_solver == 'dpm++':
        #     scheduler = FlowDPMSolverMultistepScheduler(
        #         num_train_timesteps=self.num_train_timesteps,
        #         shift=1,
        #         use_dynamic_shifting=False)
        
        # self.scheduler = scheduler
        F = frame_num
        self.unet_to_device = unet_to_device
        self.vae_to_device = vae_to_device
        
        # n_prompt = self.sample_neg_prompt
        if n_prompt == "" or n_prompt == ".":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(unet_to_device)
            context = self.text_encoder([input_prompt], unet_to_device)
            context_null = self.text_encoder([n_prompt], unet_to_device)
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            
            
        context = [t.to(unet_to_device) for t in context]
        context_null = [t.to(unet_to_device) for t in context_null]
        
        
        self.arg_c = {
            'context': [context[0]],
            'seq_len': seq_len,
        }
        self.arg_null = {
            'context': context_null,
            'seq_len': seq_len,
        }
        self.seq_len = seq_len
        
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        mask1, mask2 = masks_like([noise], zero=True)
        
        self.latent_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2])
        
        self.mask2 = mask2[0].to(unet_to_device)

        
        self.resample_timestep = resample_timestep
        if resample_timestep:
            self.sds_iterations = sds_iterations
            _, _, sampled_t = sample_from_weighted_logit_normal(mu=0.0, sigma=1.0, T=sds_iterations+1, N_grid=1000000, weight=1.0)
            sampled_t = np.clip(sampled_t, t_range[0], t_range[1])
            self.resample_ts = sampled_t
        
        
        self.frame_num = frame_num
        
        
        del self.text_encoder
        if self.off_load:
            self.model.cpu()
        else:
            self._move_model_to_forward_device()
        if del_decoder:
            self.vae.model.decoder.to("cpu")
        # self.vae.to(vae_to_device)

    def offload_model(self):
        self.model.cpu()
    
    def onload_model(self):
        self._move_model_to_forward_device()
    
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
            latents = self.vae.encode([pred_rgb], use_ckpt=True)[0]
            return latents.detach()
        
    def calc_sds_loss(self, pred_rgb, target):
        pred_rgb = pred_rgb * 2.0 - 1.0
        pred_rgb = pred_rgb.permute(1,0,2,3)
        # pred_rgb = pred_rgb[0:3]
        latents = self.vae.encode([pred_rgb], use_ckpt=True)[0]
        loss = 0.5 * F.mse_loss(latents[:,1:].float(), target[:,1:], reduction='sum')
        return loss
    
    def get_latents(self, pred_rgb):
        pred_rgb = pred_rgb * 2.0 - 1.0
        pred_rgb = pred_rgb.permute(1,0,2,3)
        # pred_rgb = pred_rgb[0:3]
        latents = self.vae.encode([pred_rgb], use_ckpt=True)[0]
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
        sds_w = 1.0,
        multistep_sample = False,
        log_run_time=False,
        use_tiny_vae=False,
        ):
        _ = use_tiny_vae
        if log_run_time:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            starter.record()
        
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
            z = self.vae.encode([ref_img])
            self.saved_z = z[0]
            z = [u.detach().to(self.unet_to_device) for u in z]
            
        
        if pass_encoder:
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb], use_ckpt=use_ckpt)[0]
        else:
            latents = pred_rgb
            
            
        batch_size = 1
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
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
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            t_ratio = t.item() / self.num_train_timesteps
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                # w = t_ratio
            self.saved_w = w
            # pred noise
            arg_c = self.arg_c
            arg_null = self.arg_null
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            timestep = t
            temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
            ])
            timestep = temp_ts.unsqueeze(0)
            
            latents_noisy = (1. - self.mask2) * z[0] + self.mask2 * latents_noisy
            if multistep_sample:
                restored_latent = self.rf_solve(latents_noisy.detach(), z[0], self.arg_c['context'], self.arg_null['context'], t_ratio, steps=5,
                                                guidance_scale=guidance_scale, inverse=False)
                noise_pred = (latents_noisy - restored_latent) / t_ratio
                noise = noise - latents
                if save_mid:
                    self.saved_pred_original = restored_latent
            else:
                if self.off_load:
                    self._move_model_to_forward_device()
                noise_pred_cond = self.model(
                        [latents_noisy], t=timestep, **arg_c)[0]
                
                noise_pred_uncond = self.model(
                    [latents_noisy], t=timestep, **arg_null)[0]
                
                noise_pred_cond = noise_pred_cond.detach().to("cuda")
                noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # noise_pred = noise_pred + latents'
                if save_mid:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    pred_original = latents_noisy - t_ratio * flow
                    pred_original = (1. - self.mask2.to("cuda")) * z[0].to("cuda") + self.mask2.to("cuda") * pred_original
                    self.saved_pred_original = pred_original.detach().clone()
                    # self.saved_latents = latents.detach().clone()
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        grad[:,0,:,:] = 0.0  # set the first channel to zero, as it is not used in the model
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            self.model.cpu()
            torch.cuda.empty_cache()
            return target

        loss = 0.5 * F.mse_loss(latents[:,1:].float(), target[:,1:], reduction='sum')
        
        if log_run_time:
            ender.record()
            torch.cuda.synchronize()
            ms = starter.elapsed_time(ender)
            print(f"[Timing] SDS step: {ms / 1000.0}")
        
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
        sds_w = 1.0,
        multistep_sample = False,
        ):
        
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
            z = self.vae.encode([ref_img])
            self.saved_z = z[0]
            z = [u.detach().to(self.unet_to_device) for u in z]
            
        
        if pass_encoder:
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb], use_ckpt=use_ckpt)[0]
        else:
            latents = pred_rgb
            
            
        batch_size = 1
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
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
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            t_ratio = t.item() / self.num_train_timesteps
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                # w = t_ratio
            self.saved_w = w
            # pred noise
            arg_c = self.arg_c
            arg_null = self.arg_null
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            timestep = t
            temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
            ])
            timestep = temp_ts.unsqueeze(0)
            
            latents_noisy = (1. - self.mask2) * z[0] + self.mask2 * latents_noisy
            if multistep_sample:
                restored_latent = self.rf_solve(latents_noisy.detach(), z[0], self.arg_c['context'], self.arg_null['context'], t_ratio, steps=5,
                                                guidance_scale=guidance_scale, inverse=False)
                noise_pred = (latents_noisy - restored_latent) / t_ratio
                noise = noise - latents
                if save_mid:
                    self.saved_pred_original = restored_latent
            else:
                noise_pred_cond = self.model(
                        [latents_noisy], t=timestep, **arg_c)[0]
                
                noise_pred_uncond = self.model(
                    [latents_noisy], t=timestep, **arg_null)[0]
                
                noise_pred_cond = noise_pred_cond.detach().to("cuda")
                noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # noise_pred = noise_pred + latents'
                if save_mid:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    pred_original = latents_noisy - t_ratio * flow
                    pred_original = (1. - self.mask2.to("cuda")) * z[0].to("cuda") + self.mask2.to("cuda") * pred_original
                    self.saved_pred_original = pred_original.detach().clone()
                    # self.saved_latents = latents.detach().clone()
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        grad[:,0,:,:] = 0.0  # set the first channel to zero, as it is not used in the model
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents[:,1:].float(), target[:,1:], reduction='sum')
        
        return loss, target
    
    def sds_step_batch(self,
        pred_rgb_list,
        step_ratio=None,
        ref_img=None,
        guidance_scale=100.0,
        pass_encoder = True,
        pred_flow = False,
        use_ckpt = True,
        save_mid = False,
        sds_w = 1.0,
        multistep_sample = False,
        ):
        
        pred_rgb_list = [u.detach() for u in pred_rgb_list]
        with torch.no_grad():
            ref_img_list = []
            for i in range(len(pred_rgb_list)):
                if ref_img==None:
                    cur_ref_img = pred_rgb_list[i][0].detach().clone()
                else:
                    cur_ref_img = ref_img[i]
                cur_ref_img = cur_ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
                ref_img_list.append(cur_ref_img)
            z = self.vae.encode(ref_img_list)
            self.saved_z = z[0]
            z = [u.detach().to(self.unet_to_device) for u in z]
            
        
        if pass_encoder:
            with torch.no_grad():
                for i in range(len(pred_rgb_list)):
                    pred_rgb_list[i] = pred_rgb_list[i] * 2.0 - 1.0
                    pred_rgb_list[i] = pred_rgb_list[i].permute(1,0,2,3)
                # pred_rgb = pred_rgb[0:3]
                latents = self.vae.encode(pred_rgb_list, use_ckpt=use_ckpt)
        else:
            latents = pred_rgb_list
            
            
        batch_size = len(latents)
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
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
            noise = [torch.randn_like(latents[i]) for i in range(batch_size)]
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            t_ratio = t[0].item() / self.num_train_timesteps
            # latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            latents_noisy = []
            for i in range(batch_size):
                cur_latents_noisy = latents[i]*(1.0-t_ratio) + t_ratio*noise[i]
                cur_latents_noisy = cur_latents_noisy.detach().to(self.unet_to_device)
                latents_noisy.append(cur_latents_noisy)
            #w = (1 - alphas[t]).item()
            w = 1.0
            if self.resample_timestep is False:
                w = w_logit_normal(t_ratio)
                # w = t_ratio
            self.saved_w = w
            # pred noise
            arg_c = self.arg_c
            arg_null = self.arg_null
            t=t.to(self.unet_to_device)
            
            timestep = t
            temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
            ])
            timestep = temp_ts.unsqueeze(0)
            
            latents_noisy = (1. - self.mask2) * z[0] + self.mask2 * latents_noisy
            if multistep_sample:
                restored_latent = self.rf_solve(latents_noisy.detach(), z[0], self.arg_c['context'], self.arg_null['context'], t_ratio, steps=5,
                                                guidance_scale=guidance_scale, inverse=False)
                noise_pred = (latents_noisy - restored_latent) / t_ratio
                noise = noise - latents
                if save_mid:
                    self.saved_pred_original = restored_latent
            else:
                noise_pred_cond = self.model(
                        [latents_noisy], t=timestep, **arg_c)[0]
                
                noise_pred_uncond = self.model(
                    [latents_noisy], t=timestep, **arg_null)[0]
                
                noise_pred_cond = noise_pred_cond.detach().to("cuda")
                noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # noise_pred = noise_pred + latents'
                if save_mid:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    pred_original = latents_noisy - t_ratio * flow
                    pred_original = (1. - self.mask2.to("cuda")) * z[0].to("cuda") + self.mask2.to("cuda") * pred_original
                    self.saved_pred_original = pred_original.detach().clone()
                    # self.saved_latents = latents.detach().clone()
                if pred_flow is False:
                    flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_noisy = latents_noisy.detach().to("cuda")
                    noise_pred = latents_noisy + (1 - t_ratio) * flow
                else:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    noise = noise - latents
                
        

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        grad[:,0,:,:] = 0.0  # set the first channel to zero, as it is not used in the model
        
        if save_mid:
            self.saved_grad = grad.detach().clone()

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents[:,1:].float(), target[:,1:], reduction='sum')
        
        return loss
    
    
    def get_uncond_predict(self, latents, step_ratio, guidance_scale, ref_img):
        
        with torch.no_grad():
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
            z = self.vae.encode([ref_img])
            self.saved_z = z[0]
            z = [u.detach().to(self.unet_to_device) for u in z]
        
        t = ((1 - step_ratio) * self.num_train_timesteps)
        t = min(max(t, self.min_step), self.max_step)
        t = torch.full((1,), t, dtype=torch.float32, device=self.device)

        self.saved_t = t
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            t_ratio = t.item() / self.num_train_timesteps
            latents_noisy = latents*(1.0-t_ratio) + t_ratio*noise
            #w = (1 - alphas[t]).item()
            w = 1.0
                # w = t_ratio
            self.saved_w = w
            # pred noise
            arg_c = self.arg_c
            arg_null = self.arg_null
            latents_noisy = latents_noisy.detach().to(self.unet_to_device)
            t=t.to(self.unet_to_device)
            
            timestep = t
            temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
            ])
            timestep = temp_ts.unsqueeze(0)
            
            latents_noisy = (1. - self.mask2) * z[0] + self.mask2 * latents_noisy

            noise_pred_cond = self.model(
                    [latents_noisy], t=timestep, **arg_c)[0]
            
            noise_pred_uncond = self.model(
                [latents_noisy], t=timestep, **arg_null)[0]
            
            noise_pred_cond = noise_pred_cond.detach().to("cuda")
            noise_pred_uncond = noise_pred_uncond.detach().to("cuda")
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # noise_pred = noise_pred + latents'
            flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents_noisy = latents_noisy.detach().to("cuda")
            pred_original = latents_noisy - t_ratio * flow
            pred_original = (1. - self.mask2.to("cuda")) * z[0].to("cuda") + self.mask2.to("cuda") * pred_original
            self.saved_pred_original = pred_original.detach().clone()
    def generate_test(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 input_prompt2=None,
                 mix_ratio=1.0):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        return self.i2v_test(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
                input_prompt2=input_prompt2,
                mix_ratio=mix_ratio)
        
    def i2v_test(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            input_prompt2=None,
            mix_ratio=1.0):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "" or n_prompt == ".":
            n_prompt = self.sample_neg_prompt

        # preprocess
        self.text_encoder.model.to(self.device)
        context1 = self.text_encoder([input_prompt], self.device)
        context2 = self.text_encoder([input_prompt2], self.device) if input_prompt2 else None
        
        if context2 is not None:
            context1 = [torch.cat([u, u.new_zeros(self.model.text_len - u.size(0),u.size(1))]) for u in context1]
            context2 = [torch.cat([u, u.new_zeros(self.model.text_len - u.size(0),u.size(1))]) for u in context2]
            context = [(1 - mix_ratio) * c2 + mix_ratio * c1 for c1, c2 in zip(context1, context2)]
        else:
            context = context1
            
        context_null = self.text_encoder([n_prompt], self.device)
        self.text_encoder.model.cpu()
        # else:
        #     context = self.text_encoder([input_prompt], torch.device('cpu'))
        #     context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        #     context = [t.to(self.device) for t in context]
        #     context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

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
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # flow = noise_pred
                # predicted_latent = latent - t/self.num_train_timesteps * flow
                # predicted_latent = (1. - mask2[0]) * z[0] + mask2[0] * predicted_latent
                # save_path = "temp_latent"
                # self.model.cpu()
                # videos = self.vae.decode([predicted_latent])
                # os.makedirs(save_path, exist_ok=True)
                # cache_video(
                #     tensor=videos[0][None],
                #     save_file=os.path.join(save_path, f"video_{t}.mp4"),
                #     fps=16,
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1))
                # self.model.to(self.device)
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
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

        return videos[0] if self.rank == 0 else None
    
    def rf_solve(self, latent, z0, prompt_embed, neg_prompt_embed, noise_ratio, 
                   steps=5, guidance_scale=1.0, flow_shift=5.0, div_steps=0.0, fixed_steps=True, inverse=False):
        true_steps = steps
        if fixed_steps:
            true_steps = int(flow_shift * steps / noise_ratio - (flow_shift - 1) * steps)
        timesteps = np.linspace(0, 0.999, true_steps)
        timesteps = flow_shift * timesteps / (1 +
                                       (flow_shift - 1) * timesteps)
        final_timesteps = []
        for i in range(timesteps.shape[0]):
            if timesteps[i] < noise_ratio:
                final_timesteps.append(
                    timesteps[i] * self.num_train_timesteps)
        final_timesteps.append(noise_ratio * self.num_train_timesteps)
        if inverse is False:
            final_timesteps = final_timesteps[::-1]
            
        def get_pred_flow(x_t, timestep):
            if guidance_scale >= 1.0:
                temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)
                input_latent = (1. - self.mask2) * z0 + self.mask2 * x_t
                arg_c = {
                    'context': prompt_embed,
                    'seq_len': self.seq_len,
                }
                arg_null = {
                    'context': neg_prompt_embed,
                    'seq_len': self.seq_len,
                }
                noise_pred_cond = self.model(
                    [input_latent], t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    [input_latent], t=timestep, **arg_null)[0]
                flow = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                return flow
            else:
                temp_ts = (self.mask2[0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(self.seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)
                input_latent = (1. - self.mask2) * z0 + self.mask2 * x_t
                arg_null = {
                    'context': neg_prompt_embed,
                    'seq_len': self.seq_len,
                }
                noise_pred_uncond = self.model(
                    [input_latent], t=timestep, **arg_null)[0]
                flow = noise_pred_uncond
                return flow
        
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            latent = (1. - self.mask2) * z0 + self.mask2 * latent
            for i in range(0, len(final_timesteps)-1):
                cur_t = final_timesteps[i] / self.num_train_timesteps
                next_t = final_timesteps[i+1] / self.num_train_timesteps
                if div_steps <= 0.0:
                    cur_div_steps = (next_t - cur_t) / 2.0
                else:
                    cur_div_steps = div_steps
                cur_flow = get_pred_flow(latent, final_timesteps[i])
                delta_latent = latent + cur_div_steps * cur_flow
                delta_latent = (1. - self.mask2) * z0 + self.mask2 * delta_latent
                second_flow = get_pred_flow(delta_latent, final_timesteps[i] + cur_div_steps * self.num_train_timesteps)
                second_div = (second_flow - cur_flow) / cur_div_steps
                latent = latent + (next_t - cur_t) * cur_flow + 0.5 * ((next_t - cur_t)**2) * second_div
                latent = (1. - self.mask2) * z0 + self.mask2 * latent
            
        return latent
    
    
    def rf_sds_step(self,
        pred_rgb,
        step_ratio=None,
        ref_img=None,
        inverse_steps = 5,
        flow_shift = 5.0,
        guidance_scale=5.0,
        pass_encoder = True,
        use_ckpt = True,
        ):
        if self.off_load:
            pred_rgb = pred_rgb.detach().clone()
        
        with torch.no_grad():
            if ref_img==None:
                ref_img = pred_rgb[0].detach().clone()
            ref_img = ref_img.sub_(0.5).div_(0.5).to(self.vae_to_device).unsqueeze(1)
            z = self.vae.encode([ref_img])
            self.saved_z = z[0]
            z = [u.detach().to(self.unet_to_device) for u in z]
            
        
        if pass_encoder:
            pred_rgb = pred_rgb * 2.0 - 1.0
            pred_rgb = pred_rgb.permute(1,0,2,3)
            # pred_rgb = pred_rgb[0:3]
            latents = self.vae.encode([pred_rgb], use_ckpt=use_ckpt)[0]
        else:
            latents = pred_rgb
            
            
        batch_size = 1
        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
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
        noise_ratio = t.item() / self.num_train_timesteps
        
        with amp.autocast(dtype=self.param_dtype), torch.no_grad():
            inversed_latent = self.rf_solve(latents.detach(), z[0], self.arg_null['context'], self.arg_null['context'], noise_ratio,
                                            inverse_steps, 1.0, flow_shift, inverse=True)
            inversed_latent = (1. - self.mask2) * z[0] + self.mask2 * inversed_latent
            # noise = (inversed_latent - latents).detach() / (noise_ratio)
            restored_latent = self.rf_solve(inversed_latent.detach(), z[0], self.arg_c['context'], self.arg_null['context'], noise_ratio,
                                            inverse_steps, guidance_scale, flow_shift, inverse=False)
            restored_latent = (1. - self.mask2) * z[0] + self.mask2 * restored_latent
            self.saved_pred_original = restored_latent
            # noise_pred = (inversed_latent - restored_latent).detach() / (noise_ratio)
            w = 1.0

        grad = w * (latents.detach() - restored_latent.detach())
        grad = torch.nan_to_num(grad)
        
        grad[:,0,:,:] = 0.0  # set the first channel to zero, as it is not used in the model

        target = (latents - grad).detach()
        
        if self.off_load:
            return target

        loss = 0.5 * F.mse_loss(latents[:,1:].float(), target[:,1:], reduction='sum')
        
        return loss
    
    def i2v_save_feat(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
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
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "" or n_prompt == ".":
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

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        cnt = 0

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

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
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
                # 'save_mid': True,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                
                cnt = cnt + 1
                if cnt == timestep_num:
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, save_mid=True, **arg_c)[0]
                    saved_mid = self.model.get_saved_mid()
                    self.model.cpu()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    return saved_mid
                else:
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, **arg_c)[0]
                
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # flow = noise_pred
                # predicted_latent = latent - t/self.num_train_timesteps * flow
                # predicted_latent = (1. - mask2[0]) * z[0] + mask2[0] * predicted_latent
                # save_path = "temp_latent"
                # self.model.cpu()
                # videos = self.vae.decode([predicted_latent])
                # os.makedirs(save_path, exist_ok=True)
                # cache_video(
                #     tensor=videos[0][None],
                #     save_file=os.path.join(save_path, f"video_{t}.mp4"),
                #     fps=16,
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1))
                # self.model.to(self.device)
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
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

        return videos[0] if self.rank == 0 else None
    
    def i2v_noise(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
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
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
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
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        # noise = noise[:,0:1].repeat(1, noise.shape[1], 1, 1)

        if n_prompt == "" or n_prompt == ".":
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

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

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
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                # flow = noise_pred
                # predicted_latent = latent - t/self.num_train_timesteps * flow
                # predicted_latent = (1. - mask2[0]) * z[0] + mask2[0] * predicted_latent
                # save_path = "temp_latent"
                # self.model.cpu()
                # videos = self.vae.decode([predicted_latent])
                # os.makedirs(save_path, exist_ok=True)
                # cache_video(
                #     tensor=videos[0][None],
                #     save_file=os.path.join(save_path, f"video_{t}.mp4"),
                #     fps=16,
                #     nrow=1,
                #     normalize=True,
                #     value_range=(-1, 1))
                # self.model.to(self.device)
                
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
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

        return videos[0] if self.rank == 0 else None
