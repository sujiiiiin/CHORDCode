import logging
import os
import types

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image

import lightx2v_platform.set_ai_device  # noqa: F401
from lightx2v.models.input_encoders.hf.wan.t5.model import T5EncoderModel
from lightx2v.models.networks.wan.distill_model import WanDistillModel
from lightx2v.models.runners.wan.wan_distill_runner import MultiDistillModelStruct
from lightx2v.models.schedulers.wan.step_distill.scheduler import Wan22StepDistillScheduler
from lightx2v.models.video_encoders.hf.wan.vae import WanVAE
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import find_torch_model_path, seed_all
from lightx2v_platform.base.global_var import AI_DEVICE


class LightX2VWan22MoeI2VDistill:
    def __init__(
        self,
        model_path,
        config_json=None,
        device_id=0,
        seed=42,
        is_attn2=False,
    ):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.config = self._load_config(model_path, config_json, is_attn2)
        self.init_device = torch.device("cpu") if self.config.get("cpu_offload", False) else torch.device(AI_DEVICE)
        self.vae_dtype = torch.float32

        self.text_encoder = self._load_text_encoder()
        self.vae_encoder, self.vae_decoder = self._load_vae()
        self.scheduler = Wan22StepDistillScheduler(self.config)
        self.model = self._load_dit_models()

        self.model.set_scheduler(self.scheduler)

    def _load_config(self, model_path, config_json, is_attn2=False):
        if config_json is None:
            if is_attn2:
                config_json = os.path.join(os.path.dirname(__file__), "configs", "wan22", "wan_moe_i2v_distill_attn2.json")
            else:
                config_json = os.path.join(os.path.dirname(__file__), "configs", "wan22", "wan_moe_i2v_distill.json")
        args = types.SimpleNamespace(
            model_cls="wan2.2_moe_distill",
            task="i2v",
            model_path=model_path,
            config_json=config_json,
            seed=0,
            prompt="",
            negative_prompt="",
            image_path="",
            save_result_path="",
            return_result_tensor=False,
        )
        return set_config(args)

    def _load_text_encoder(self):
        t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload"))
        t5_device = torch.device("cpu") if t5_offload else torch.device(AI_DEVICE)
        tokenizer_path = os.path.join(self.config["model_path"], "google/umt5-xxl")
        t5_quantized = self.config.get("t5_quantized", False)
        if t5_quantized:
            t5_quant_scheme = self.config.get("t5_quant_scheme", None)
            if t5_quant_scheme is None:
                raise ValueError("t5_quant_scheme is required when t5_quantized is enabled.")
            tmp_t5_quant_scheme = t5_quant_scheme.split("-")[0]
            t5_model_name = f"models_t5_umt5-xxl-enc-{tmp_t5_quant_scheme}.pth"
            t5_quantized_ckpt = find_torch_model_path(self.config, "t5_quantized_ckpt", t5_model_name)
            t5_original_ckpt = None
        else:
            t5_quant_scheme = None
            t5_quantized_ckpt = None
            t5_model_name = "models_t5_umt5-xxl-enc-bf16.pth"
            t5_original_ckpt = find_torch_model_path(self.config, "t5_original_ckpt", t5_model_name)

        return T5EncoderModel(
            text_len=self.config["text_len"],
            dtype=torch.bfloat16,
            device=t5_device,
            checkpoint_path=t5_original_ckpt,
            tokenizer_path=tokenizer_path,
            shard_fn=None,
            cpu_offload=t5_offload,
            t5_quantized=t5_quantized,
            t5_quantized_ckpt=t5_quantized_ckpt,
            quant_scheme=t5_quant_scheme,
            load_from_rank0=self.config.get("load_from_rank0", False),
            lazy_load=self.config.get("t5_lazy_load", False),
        )

    def _load_vae(self):
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        vae_device = torch.device("cpu") if vae_offload else torch.device(AI_DEVICE)
        vae_name = self.config.get("vae_name", "Wan2.1_VAE.pth")
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", vae_name),
            "device": vae_device,
            "parallel": False,
            "use_tiling": self.config.get("use_tiling_vae", False),
            "cpu_offload": vae_offload,
            "dtype": self.vae_dtype,
            "load_from_rank0": self.config.get("load_from_rank0", False),
            "use_lightvae": self.config.get("use_lightvae", False),
        }

        if self.config["task"] not in ["i2v", "flf2v", "animate", "vace", "s2v"]:
            return None, None

        vae_encoder = WanVAE(**vae_config)
        vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    def _load_dit_models(self):
        high_noise_model_path = self._resolve_high_noise_path()
        low_noise_model_path = self._resolve_low_noise_path()

        high_noise_model = WanDistillModel(
            high_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_high_noise",
        )
        low_noise_model = WanDistillModel(
            low_noise_model_path,
            self.config,
            self.init_device,
            model_type="wan2.2_moe_low_noise",
        )
        return MultiDistillModelStruct([high_noise_model, low_noise_model], self.config, self.config["boundary_step_index"])

    def _resolve_high_noise_path(self):
        if self.config.get("dit_quantized", False) and self.config.get("high_noise_quantized_ckpt", None):
            return self.config["high_noise_quantized_ckpt"]
        if self.config.get("high_noise_original_ckpt", None):
            return self.config["high_noise_original_ckpt"]

        high_noise_model_path = os.path.join(self.config["model_path"], "high_noise_model")
        if os.path.isdir(high_noise_model_path):
            return high_noise_model_path
        high_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "high_noise_model")
        if os.path.isdir(high_noise_model_path):
            return high_noise_model_path
        raise FileNotFoundError("High noise model does not exist in model_path.")

    def _resolve_low_noise_path(self):
        if self.config.get("dit_quantized", False) and self.config.get("low_noise_quantized_ckpt", None):
            return self.config["low_noise_quantized_ckpt"]
        if not self.config.get("dit_quantized", False) and self.config.get("low_noise_original_ckpt", None):
            return self.config["low_noise_original_ckpt"]

        low_noise_model_path = os.path.join(self.config["model_path"], "low_noise_model")
        if os.path.isdir(low_noise_model_path):
            return low_noise_model_path
        low_noise_model_path = os.path.join(self.config["model_path"], "distill_models", "low_noise_model")
        if os.path.isdir(low_noise_model_path):
            return low_noise_model_path
        raise FileNotFoundError("Low noise model does not exist in model_path.")

    def encode_prompt(self, prompt, negative_prompt=""):
        with torch.no_grad():
            context = self.text_encoder.infer([prompt])
            context = torch.stack(
                [
                    torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))])
                    for u in context
                ]
            )
            if self.config.get("enable_cfg", False):
                context_null = self.text_encoder.infer([negative_prompt])
                context_null = torch.stack(
                    [
                        torch.cat([u, u.new_zeros(self.config["text_len"] - u.size(0), u.size(1))])
                        for u in context_null
                    ]
                )
            else:
                context_null = None
        return {"context": context, "context_null": context_null}

    def read_image(self, image):
        if torch.is_tensor(image):
            img_tensor = image
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor.to(self.init_device), None

        if isinstance(image, Image.Image):
            img_ori = image
        else:
            img_ori = Image.open(image).convert("RGB")
        img_tensor = TF.to_tensor(img_ori).sub_(0.5).div_(0.5).unsqueeze(0).to(self.init_device)
        return img_tensor, img_ori

    def encode_image(self, image):
        img_tensor, _ = self.read_image(image)
        vae_encoder_out, latent_shape = self._run_vae_encoder(img_tensor)
        return vae_encoder_out, latent_shape

    def build_condition(self, text_encoder_output, vae_encoder_out, clip_encoder_out=None):
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out,
            },
        }

    def prepare_latents(self, latent_shape, seed=None):
        if seed is None:
            seed = self.seed
        seed_all(seed)
        self.scheduler.prepare(seed, latent_shape)
        return self.scheduler.latents

    def dit_forward(self, latents_noisy, t, text_encoder_output, image_encoder_output, step_index=None):
        self.scheduler.latents = latents_noisy
        if step_index is not None:
            self.scheduler.step_index = step_index
        self.scheduler.timestep_input = t
        inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }
        self.model.infer(inputs)
        return self.scheduler.noise_pred

    def sample_latents(self, text_encoder_output, vae_encoder_out, latent_shape, seed=None):
        latents = self.prepare_latents(latent_shape, seed=seed)
        arg_c = self.build_condition(text_encoder_output, vae_encoder_out)

        for step_index in range(self.scheduler.infer_steps):
            self.scheduler.step_pre(step_index)
            t = self.scheduler.timestep_input
            _ = self.dit_forward(latents, t=t, step_index=step_index, **arg_c)
            self.scheduler.step_post()
            latents = self.scheduler.latents
        return latents

    def decode(self, latents):
        if self.vae_decoder is None:
            raise RuntimeError("VAE decoder is not initialized.")
        return self.vae_decoder.decode(latents.to(self.vae_dtype))

    def generate(self, image, prompt, negative_prompt="", seed=None):
        logging.info("Encoding prompt.")
        text_encoder_output = self.encode_prompt(prompt, negative_prompt=negative_prompt)
        logging.info("Encoding image.")
        vae_encoder_out, latent_shape = self.encode_image(image)
        logging.info("Sampling latents.")
        latents = self.sample_latents(text_encoder_output, vae_encoder_out, latent_shape, seed=seed)
        logging.info("Decoding latents.")
        return self.decode(latents)

    def _adjust_latent_for_grid_splitting(self, latent_h, latent_w, world_size):
        world_size_h, world_size_w = 1, 1
        if world_size <= 1:
            return latent_h, latent_w, world_size_h, world_size_w

        priority_grids = []
        if world_size == 8:
            priority_grids = [(2, 4), (4, 2), (1, 8), (8, 1)]
        elif world_size == 4:
            priority_grids = [(2, 2), (1, 4), (4, 1)]
        elif world_size == 2:
            priority_grids = [(1, 2), (2, 1)]
        else:
            for h in range(1, int(np.sqrt(world_size)) + 1):
                if world_size % h == 0:
                    w = world_size // h
                    priority_grids.append((h, w))

        for world_size_h, world_size_w in priority_grids:
            if latent_h % world_size_h == 0 and latent_w % world_size_w == 0:
                return latent_h, latent_w, world_size_h, world_size_w

        best_grid = (1, world_size)
        min_total_padding = float("inf")

        for world_size_h, world_size_w in priority_grids:
            pad_h = (world_size_h - (latent_h % world_size_h)) % world_size_h
            pad_w = (world_size_w - (latent_w % world_size_w)) % world_size_w
            total_padding = pad_h + pad_w
            if total_padding < min_total_padding:
                min_total_padding = total_padding
                best_grid = (world_size_h, world_size_w)

        world_size_h, world_size_w = best_grid
        pad_h = (world_size_h - (latent_h % world_size_h)) % world_size_h
        pad_w = (world_size_w - (latent_w % world_size_w)) % world_size_w

        return latent_h + pad_h, latent_w + pad_w, world_size_h, world_size_w

    def _run_vae_encoder(self, first_frame, last_frame=None):
        if self.config.get("resize_mode", None) is None:
            h, w = first_frame.shape[2:]
            aspect_ratio = h / w
            max_area = self.config["target_height"] * self.config["target_width"]

            ori_latent_h = round(
                np.sqrt(max_area * aspect_ratio)
                // self.config["vae_stride"][1]
                // self.config["patch_size"][1]
                * self.config["patch_size"][1]
            )
            ori_latent_w = round(
                np.sqrt(max_area / aspect_ratio)
                // self.config["vae_stride"][2]
                // self.config["patch_size"][2]
                * self.config["patch_size"][2]
            )

            if dist.is_initialized() and dist.get_world_size() > 1:
                latent_h, latent_w, world_size_h, world_size_w = self._adjust_latent_for_grid_splitting(
                    ori_latent_h, ori_latent_w, dist.get_world_size()
                )
            else:
                latent_h, latent_w = ori_latent_h, ori_latent_w
                world_size_h, world_size_w = None, None

            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)
        else:
            raise NotImplementedError("resize_mode is not supported in this wrapper.")

        vae_encoder_out = self._get_vae_encoder_output(first_frame, latent_h, latent_w, last_frame, world_size_h=world_size_h, world_size_w=world_size_w)
        return vae_encoder_out, latent_shape

    def _get_vae_encoder_output(self, first_frame, lat_h, lat_w, last_frame=None, world_size_h=None, world_size_w=None):
        h = lat_h * self.config["vae_stride"][1]
        w = lat_w * self.config["vae_stride"][2]
        msk = torch.ones(
            1,
            self.config["target_video_length"],
            lat_h,
            lat_w,
            device=torch.device(AI_DEVICE),
        )
        if last_frame is not None:
            msk[:, 1:-1] = 0
        else:
            msk[:, 1:] = 0

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if last_frame is not None:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 2, h, w),
                    torch.nn.functional.interpolate(last_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                ],
                dim=1,
            ).to(AI_DEVICE)
        else:
            vae_input = torch.concat(
                [
                    torch.nn.functional.interpolate(first_frame.cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, self.config["target_video_length"] - 1, h, w),
                ],
                dim=1,
            ).to(AI_DEVICE)

        vae_encoder_out = self.vae_encoder.encode(
            vae_input.unsqueeze(0).to(self.vae_dtype),
            world_size_h=world_size_h,
            world_size_w=world_size_w,
        )
        vae_encoder_out = torch.concat([msk, vae_encoder_out]).to(self.vae_dtype)
        return vae_encoder_out

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        return [
            self.config.get("num_channels_latents", 16),
            (self.config["target_video_length"] - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]
