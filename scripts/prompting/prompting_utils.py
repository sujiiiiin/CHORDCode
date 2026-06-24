import base64
import copy
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams
from gaussian_renderer.dynamic_renderer import render_dynamic
from scene import GaussianModel, Scene
from scene.dynamic_scene import DynamicGaussianScene
from utils.orbit_cam_utils import OrbitCamera
from utils.render_utils import build_view
from utils.scene_object_utils import resolve_scene_obj_num


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def sanitize_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (text or "item")[:max_len]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"yes", "true", "t", "1", "y"}:
        return True
    if value in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def normalize_obj_names(values: Optional[Iterable[str]]) -> set[str]:
    normalized = set()
    for value in values or []:
        value = str(value).strip()
        if value:
            normalized.add(value if value.startswith("obj_") else f"obj_{value}")
    return normalized


def extract_json_block(text: str):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from VLM response: {text}")


def _clean_prompt_candidate(text: str) -> str:
    text = re.sub(r"^\s*\*\*(.*?)\*\*\s*:?", r"\1:", text.strip())
    text = text.replace("**", "").strip()
    text = text.strip("\"'` ")
    return text.strip()


def _parse_prompt_list_fallback(text: str) -> list[str]:
    numbered = []
    bullets = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?P<prefix>(?:[-*])|(?:\d+[.)]))\s+(?P<body>.+?)\s*$", line)
        if not match:
            continue
        candidate = _clean_prompt_candidate(match.group("body"))
        if not candidate:
            continue
        if match.group("prefix") in {"-", "*"}:
            bullets.append(candidate)
        else:
            numbered.append(candidate)
    return bullets or numbered


def parse_prompt_list(text: str) -> list[str]:
    try:
        payload = extract_json_block(text)
    except ValueError:
        prompts = _parse_prompt_list_fallback(text)
        if prompts:
            return prompts
        raise
    if isinstance(payload, dict):
        prompts = payload.get("prompts", [])
    elif isinstance(payload, list):
        prompts = payload
    else:
        prompts = []
    prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not prompts:
        raise ValueError(f"No prompts found in response: {text}")
    return prompts


def parse_requirement_list(text: str) -> list[str]:
    payload = extract_json_block(text)
    if isinstance(payload, dict):
        requirements = payload.get("requirements", [])
    elif isinstance(payload, list):
        requirements = payload
    else:
        requirements = []

    cleaned = []
    for requirement in requirements:
        if isinstance(requirement, dict):
            requirement = requirement.get("requirement", "")
        requirement = str(requirement).strip()
        if requirement:
            cleaned.append(requirement)
    if not cleaned:
        raise ValueError(f"No requirements found in response: {text}")
    return cleaned[:5]


def _normalize_score(value) -> float:
    try:
        score = float(value)
    except Exception:
        score = 0.0
    if score > 1.0 and score <= 10.0:
        score /= 10.0
    elif score > 10.0 and score <= 100.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


def _normalize_requirement_scores(values) -> list[dict]:
    normalized = []
    for item in values or []:
        if isinstance(item, str):
            normalized.append(
                {
                    "requirement": item,
                    "score": 0.0,
                    "visible": False,
                    "reason": "",
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        requirement = str(item.get("requirement", "")).strip()
        score = _normalize_score(item.get("score", item.get("visibility_score", 0.0)))
        visible = item.get("visible", item.get("satisfied", score >= 0.75))
        if isinstance(visible, str):
            visible = visible.strip().lower() in {"true", "yes", "1"}
        normalized.append(
            {
                "requirement": requirement,
                "score": score,
                "visible": bool(visible),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return normalized


def _normalize_string_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    return [str(value).strip() for value in values if str(value).strip()]


def parse_judgment(text: str) -> dict:
    payload = extract_json_block(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected judgment object, got: {text}")
    raw_payload = copy.deepcopy(payload)
    motion_score = _normalize_score(
        payload.get("motion_adherence_score", payload.get("motion_score", payload.get("score", 0.0)))
    )
    quality_score = _normalize_score(
        payload.get("video_quality_score", payload.get("quality_score", payload.get("score", 0.0)))
    )
    requirement_scores = _normalize_requirement_scores(payload.get("requirement_scores", []))
    missing_requirements = _normalize_string_list(
        payload.get(
            "missing_or_uncertain_requirements",
            payload.get("missing_requirements", []),
        )
    )
    quality_issues = _normalize_string_list(payload.get("quality_issues", []))
    return {
        "motion_adherence_score": motion_score,
        "video_quality_score": quality_score,
        "combined_score": min(motion_score, quality_score),
        "requirement_scores": requirement_scores,
        "missing_or_uncertain_requirements": missing_requirements,
        "quality_issues": quality_issues,
        "observed_motion": str(payload.get("observed_motion", "")).strip(),
        "reason": str(payload.get("reason", "")).strip(),
        "raw_judgment": raw_payload,
    }


def add_scene_args(parser):
    lp = ModelParams(parser)
    parser.add_argument("--obj_num", type=int, default=-1)
    parser.add_argument("--static_id", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    return lp


def add_render_args(parser):
    parser.add_argument("--azims", nargs="+", type=float, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--azim_l", type=float, default=0.0)
    parser.add_argument("--azim_r", type=float, default=360.0)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--cam_radius", type=float, default=2.0)
    parser.add_argument("--cam_height", type=float, default=0.0)
    parser.add_argument("--no_bg", action="store_true", default=False)
    parser.add_argument("--exclude_objs", nargs="+", type=str, default=[])
    parser.add_argument("--bg_color", choices=["white", "black"], default="white")


def add_vlm_args(parser):
    parser.add_argument("--vlm", choices=["gpt", "gemini", "qwen_local"], default="qwen_local")
    parser.add_argument("--gpt_model", default="gpt-4.1")
    parser.add_argument("--gemini_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--gemini_poll_sec", type=float, default=5.0)
    parser.add_argument("--qwen_model_path", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--qwen_device", default="cuda")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=512)
    parser.add_argument("--video_judge_frame_count", type=int, default=8)


def add_video_generation_args(parser):
    parser.add_argument("--use_original_wan", action="store_true", default=False)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--sample_solver", default="unipc")
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--is_attn2", action="store_true", default=False)
    parser.add_argument("--base_seed", type=int, default=-1)


def resolve_azims(args) -> list[float]:
    if args.azims:
        return [float(value) for value in args.azims]
    if args.num_views <= 0:
        raise ValueError("--num_views must be positive when --azims is not provided")
    return np.linspace(args.azim_l, args.azim_r, args.num_views, endpoint=False).tolist()


def load_static_gaussian_scene(dataset, obj_num: int, static_id: int, no_bg: bool, exclude_objs: Sequence[str]):
    opt_stub = type("Opt", (), {"obj_num": obj_num, "static_id": static_id})()
    resolve_scene_obj_num(opt_stub, dataset.mesh_source_path)
    excluded = normalize_obj_names(exclude_objs)
    dynamic_scene = DynamicGaussianScene(total_time=1)

    for obj_idx in range(opt_stub.obj_num):
        obj_name = f"obj_{obj_idx}"
        if obj_name in excluded:
            print(f"[PromptTools] Skipping excluded object {obj_name}")
            continue
        if no_bg and obj_idx == static_id:
            print(f"[PromptTools] Skipping static background {obj_name}")
            continue

        cur_dataset = copy.deepcopy(dataset)
        cur_dataset.model_path = os.path.join(dataset.model_path, obj_name)
        gaussians = GaussianModel(dataset.sh_degree)
        Scene(cur_dataset, gaussians, load_iteration=-1)
        dynamic_scene.add_gaussians(gaussians, is_static=True, name=obj_name)

    if not dynamic_scene.dynamic_gaussians:
        raise ValueError("No Gaussians were loaded after applying exclusions.")
    return dynamic_scene, opt_stub.obj_num


def apply_video_render_size(args, dataset):
    from wan.configs import SIZE_CONFIGS

    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {args.size}; expected one of {sorted(SIZE_CONFIGS)}")
    dataset.image_width, dataset.image_height = SIZE_CONFIGS[args.size]
    return dataset.image_width, dataset.image_height


def resolve_video_generation_args(args):
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {args.size}; expected one of {sorted(SIZE_CONFIGS)}")

    cfg = WAN_CONFIGS["i2v-A14B"]
    if args.ckpt_dir is None:
        args.ckpt_dir = "./Wan2.2-I2V-A14B" if args.use_original_wan else "./LightX2VModel"

    if args.use_original_wan:
        args.ckpt_dir = str(validate_original_wan_i2v_model_dir(args.ckpt_dir))
        if args.sample_steps is None:
            args.sample_steps = cfg.sample_steps
    else:
        args.ckpt_dir = str(resolve_lightx2v_model_dir(args.ckpt_dir))
        if args.sample_steps is None:
            args.sample_steps = 4

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    return args


def resolve_lightx2v_model_dir(model_dir, fallback_dir="./LightX2VModel") -> Path:
    model_path = Path(model_dir)
    try:
        return validate_lightx2v_model_dir(model_path)
    except FileNotFoundError as original_error:
        fallback_path = Path(fallback_dir)
        if fallback_path == model_path:
            raise

        looks_like_full_wan = (
            (model_path / "high_noise_model" / "config.json").exists()
            and (model_path / "low_noise_model" / "config.json").exists()
            and not (model_path / "config.json").exists()
        )
        if looks_like_full_wan and fallback_path.exists():
            resolved_fallback = validate_lightx2v_model_dir(fallback_path)
            print(
                "[PromptTools] --ckpt_dir points to a full Wan checkpoint. "
                f"Prompt evaluation uses the LightX2V step-distilled model; using {resolved_fallback}.",
                flush=True,
            )
            return resolved_fallback

        raise original_error


def render_multiview_gaussian_stills(args, dataset, output_dir: Path) -> list[dict]:
    ensure_dir(output_dir)
    torch.cuda.set_device(args.device)
    dynamic_scene, resolved_obj_num = load_static_gaussian_scene(
        dataset,
        args.obj_num,
        args.static_id,
        args.no_bg,
        args.exclude_objs,
    )

    default_cam = OrbitCamera(
        dataset.image_width,
        dataset.image_height,
        r=args.cam_radius,
        fovy=dataset.fovy,
    )
    bg_value = 1.0 if args.bg_color == "white" else 0.0
    bg = torch.tensor([bg_value, bg_value, bg_value], dtype=torch.float32, device="cuda")
    azims = resolve_azims(args)
    look_at = np.array([0.0, args.cam_height, 0.0], dtype=np.float32)
    records = []

    with torch.no_grad():
        for idx, azim in enumerate(azims):
            viewmat, opencv_k = build_view(
                dataset,
                default_cam,
                args.cam_radius,
                args.elev,
                float(azim),
                look_at=look_at,
            )
            frame = render_dynamic(
                viewmat,
                opencv_k,
                dynamic_scene,
                dataset.image_width,
                dataset.image_height,
                None,
                bg,
                time=0,
            )
            image_path = output_dir / f"view_{idx:03d}_azim_{azim:g}_elev_{args.elev:g}.png"
            torchvision.utils.save_image(frame.detach().cpu(), image_path)
            records.append(
                {
                    "index": idx,
                    "azim": float(azim),
                    "elev": float(args.elev),
                    "image": str(image_path),
                }
            )

    write_json(
        output_dir / "views.json",
        {
            "resolved_obj_num": resolved_obj_num,
            "image_width": dataset.image_width,
            "image_height": dataset.image_height,
            "views": records,
        },
    )
    return records


def image_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{data}"


def upload_file_and_wait(client, file_path: Path, poll_sec: float, quiet: bool = False):
    uploaded = client.files.upload(file=str(file_path))
    while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
        if not quiet:
            print(f"[VLM] Waiting for Gemini file processing: {file_path}")
        time.sleep(poll_sec)
        uploaded = client.files.get(name=uploaded.name)
    if getattr(uploaded, "state", None) and uploaded.state.name not in {"ACTIVE", "SUCCEEDED"}:
        raise RuntimeError(f"Gemini file did not become active: {file_path} -> {uploaded.state}")
    return uploaded


def sample_video_frames(video_path: Path, max_frames: int) -> list[Image.Image]:
    try:
        import imageio.v3 as iio

        frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(video_path)]
    except Exception:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "Video-frame VLM judging requires imageio-ffmpeg or PyAV. "
                "Install one of them or use --judge manual."
            ) from exc
        container = av.open(str(video_path))
        try:
            frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
        finally:
            container.close()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[int(idx)] for idx in indices]


class VLMClient:
    def __init__(self, args):
        self.args = args
        self.provider = args.vlm
        self._client = None
        self._qwen_model = None
        self._qwen_processor = None

    def generate_json(
        self,
        instruction: str,
        image_paths: Optional[Sequence[Path]] = None,
        video_path: Optional[Path] = None,
    ) -> str:
        if self.provider == "gemini":
            return self._generate_gemini(instruction, image_paths or [], video_path)
        if self.provider == "gpt":
            return self._generate_gpt(instruction, image_paths or [], video_path)
        if self.provider == "qwen_local":
            return self._generate_qwen(instruction, image_paths or [], video_path)
        raise ValueError(f"Unknown VLM provider: {self.provider}")

    def _generate_gemini(self, instruction: str, image_paths: Sequence[Path], video_path: Optional[Path]) -> str:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError("google-genai is required for --vlm gemini") from exc
            self._client = genai.Client()
        uploads = []
        if video_path is not None:
            uploads.append(upload_file_and_wait(self._client, video_path, self.args.gemini_poll_sec))
        uploads.extend(
            upload_file_and_wait(self._client, Path(path), self.args.gemini_poll_sec)
            for path in image_paths
        )
        response = self._client.models.generate_content(
            model=self.args.gemini_model,
            contents=[instruction] + uploads,
        )
        return response.text.strip()

    def _generate_gpt(self, instruction: str, image_paths: Sequence[Path], video_path: Optional[Path]) -> str:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("openai is required for --vlm gpt") from exc
            self._client = OpenAI()

        content = [{"type": "text", "text": instruction}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(Path(path))}})
        if video_path is not None:
            content.append({"type": "text", "text": "The following images are uniformly sampled frames from the generated video."})
            for frame in sample_video_frames(Path(video_path), self.args.video_judge_frame_count):
                frame_path = Path("/tmp") / f"chord_vlm_frame_{os.getpid()}_{random.randint(0, 10**9)}.png"
                frame.save(frame_path)
                try:
                    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(frame_path)}})
                finally:
                    frame_path.unlink(missing_ok=True)

        response = self._client.chat.completions.create(
            model=self.args.gpt_model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _load_qwen(self):
        if self._qwen_model is not None:
            return
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        model_path = self.args.qwen_model_path
        print(f"[VLM] Loading local Qwen model: {model_path}", flush=True)
        self._qwen_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._qwen_model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto" if self.args.qwen_device == "auto" else None,
            trust_remote_code=True,
        )
        if self.args.qwen_device != "auto":
            self._qwen_model = self._qwen_model.to(self.args.qwen_device)
        self._qwen_model.eval()
        print("[VLM] Local Qwen model loaded", flush=True)

    def _qwen_input_device(self):
        if hasattr(self._qwen_model, "device"):
            return self._qwen_model.device
        return next(self._qwen_model.parameters()).device

    def _generate_qwen(self, instruction: str, image_paths: Sequence[Path], video_path: Optional[Path]) -> str:
        self._load_qwen()
        temp_paths = []
        content = [
            {"type": "image", "url": str(Path(path).resolve())}
            for path in image_paths
        ]
        if video_path is not None:
            content.append(
                {
                    "type": "text",
                    "text": "The following images are uniformly sampled frames from the generated video.",
                }
            )
            for frame in sample_video_frames(Path(video_path), self.args.video_judge_frame_count):
                frame_path = Path("/tmp") / f"chord_qwen_frame_{os.getpid()}_{random.randint(0, 10**9)}.png"
                frame.save(frame_path)
                temp_paths.append(frame_path)
                content.append({"type": "image", "url": str(frame_path.resolve())})

        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        inputs = self._qwen_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_device = self._qwen_input_device()
        inputs = inputs.to(input_device) if hasattr(inputs, "to") else {
            key: value.to(input_device) for key, value in inputs.items()
        }
        try:
            print("[VLM] Running local Qwen generation", flush=True)
            with torch.no_grad():
                generated = self._qwen_model.generate(
                    **inputs,
                    max_new_tokens=self.args.qwen_max_new_tokens,
                    do_sample=False,
                )
            print("[VLM] Local Qwen generation finished", flush=True)
            prompt_len = inputs["input_ids"].shape[-1]
            output_ids = generated[:, prompt_len:]
            return self._qwen_processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)


def build_prompt_suggestion_instruction(num_prompts: int) -> str:
    return (
        "You are helping choose prompts for dynamic 3D scene optimization. "
        "You will see multiview renders of a static 3D scene represented by fitted Gaussians. "
        f"Generate {num_prompts} concise, feasible image-to-video motion prompts for the visible scene. "
        "The prompts should describe physically plausible object motion that a video generation model can show clearly. "
        "Avoid tiny hand/finger actions, text, invisible causes, large camera motion, and actions requiring objects not visible in the scene. "
        "Prefer clear coarse interactions, posture changes, object rotations, pushes, falls, openings, or simple articulated motion. "
        "Keep object identities and scene geometry consistent. "
        "Return strict JSON only in the format {\"prompts\": [\"prompt 1\", \"prompt 2\"]}."
    )


def build_prompt_expansion_instruction(prompt: str, num_expansions: int) -> str:
    return (
        "You are expanding a short motion prompt into video-generation prompts. "
        "You will see multiview renders of the scene. "
        "Preserve the original action, actors, objects, and final outcome, but add useful physical detail that may help an image-to-video model. "
        "Describe concrete intermediate steps, contact, weight transfer, object state changes, and causal motion when they are relevant. "
        "Keep each prompt moderate in length: 1-2 sentences, roughly 25-55 words. "
        "Do not add new objects, change the action, add camera movement, or write a long environmental description. "
        "Examples of the desired level of detail: "
        "\"The child jumps on the seesaw, sending the wooden brick flying.\" can become "
        "\"Walk the child to the nearer seat of the seesaw, have them grasp the handle, turn, and sit down slowly. "
        "As their weight settles, the beam pivots toward the child, the far platform rises, the brick loses grip, slides to the outer edge, tips, and falls to the ground.\" "
        "\"The eagle lands on the man's arm.\" can become "
        "\"The man extends his forearm horizontally and steady, inviting the eagle to glide in, flare its wings, lower its talons, and perch.\" "
        "\"The man pets the dog.\" can become "
        "\"The man bends at the waist, leans his upper body down toward the dog, and rubs the dog's head while the tail thumps in response.\" "
        "\"The hand presses the lamp downward.\" can become "
        "\"The hand steadies the lamp stem, then presses the lamp head downward until it settles into a low task-light angle.\" "
        "\"The robot arm picks up the wooden block.\" can become "
        "\"The robot arm centers above the wooden block, opens its gripper, lowers straight down, pinches the block, and lifts it to a hover.\" "
        "\"The cat walks across the cushion.\" can become "
        "\"The cat strolls across the cushion, each paw dimpling the fabric under its weight before the cushion relaxes back after every step.\" "
        f"Original prompt: {prompt}\n"
        f"Return {num_expansions} expanded prompts as strict JSON only in the format "
        "{\"prompts\": [\"expanded prompt 1\", \"expanded prompt 2\"]}."
    )


def build_target_requirements_instruction(prompt: str) -> str:
    return (
        "Break the target video prompt into visible motion requirements for judging generated videos. "
        "Each requirement should be directly checkable from the video. "
        "Focus on actors, objects, required motion events, causal order, and final outcome. "
        "Do not include video quality, style, camera motion, or hidden intent. "
        "Use 2-5 short requirements. "
        f"Target prompt: {prompt}\n"
        "Return strict JSON only in the format "
        "{\"requirements\": [\"visible requirement 1\", \"visible requirement 2\"]}."
    )


def build_video_judgment_instruction(prompt: str, requirements: Sequence[str]) -> str:
    requirements_text = "\n".join(f"- {requirement}" for requirement in requirements)
    return (
        "You are judging a generated image-to-video result. Judge only what is visible in the video; "
        "do not infer that an event happened if it is absent, occluded, or only implied.\n"
        f"Target prompt: {prompt}\n"
        f"Visible motion requirements:\n{requirements_text}\n"
        "Give two independent scores from 0.0 to 1.0.\n"
        "Motion adherence score: "
        "1.0 means all required motion events, causal order, and requested outcome are clearly visible; "
        "0.75 means the main action and outcome are visible with only minor ambiguity or occlusion; "
        "0.5 means a partial match where some required events appear but a major event, outcome, or causal link is weak or missing; "
        "0.25 means only weak hints of the requested motion are present; "
        "0.0 means the requested motion is absent, contradicted, or replaced by a different action.\n"
        "Video quality score: "
        "1.0 means identities and objects stay stable, motion is temporally coherent and physically plausible, and there are no distracting artifacts; "
        "0.75 means the video is usable with minor flicker, small geometry issues, or mild physical oddities; "
        "0.5 means the video is understandable but has noticeable identity drift, object deformation, flicker, or awkward physics; "
        "0.25 means severe artifacts, object disappearance, large identity changes, or implausible motion make it hard to use; "
        "0.0 means the video is broken or unusable.\n"
        "Return strict JSON only in this format: "
        "{\"motion_adherence_score\": 0.0, "
        "\"video_quality_score\": 0.0, "
        "\"requirement_scores\": ["
        "{\"requirement\": \"requirement text\", \"score\": 0.0, \"visible\": false, \"reason\": \"short reason\"}"
        "], "
        "\"missing_or_uncertain_requirements\": [\"requirement text\"], "
        "\"quality_issues\": [\"issue\"], "
        "\"observed_motion\": \"short description of visible motion\", "
        "\"reason\": \"short overall reason\"}."
    )


def create_video_pipeline(args):
    if args.use_original_wan:
        return _create_original_wan_pipeline(args)
    return _create_lightx2v_pipeline(args)


def _create_lightx2v_pipeline(args):
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from lightx2v.chord_adapter.image2video import WanI2V

    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {args.size}; expected one of {sorted(SIZE_CONFIGS)}")
    args.ckpt_dir = str(validate_lightx2v_model_dir(args.ckpt_dir))
    cfg = WAN_CONFIGS["i2v-A14B"]

    print("[PromptTools] Loading LightX2V step-distilled I2V pipeline")
    return WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        off_load=args.offload_model,
        is_attn2=args.is_attn2,
    ), cfg


def _create_original_wan_pipeline(args):
    from wan import WanI2V
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {args.size}; expected one of {sorted(SIZE_CONFIGS)}")
    args.ckpt_dir = str(validate_original_wan_i2v_model_dir(args.ckpt_dir))
    cfg = WAN_CONFIGS["i2v-A14B"]

    print("[PromptTools] Loading original Wan2.2 I2V pipeline")
    return WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    ), cfg


def validate_lightx2v_model_dir(model_dir) -> Path:
    model_path = Path(model_dir)
    required_paths = [
        model_path / "config.json",
        model_path / "Wan2.1_VAE.pth",
        model_path / "models_t5_umt5-xxl-enc-bf16.pth",
        model_path / "high_noise_model",
        model_path / "low_noise_model",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "LightX2V step-distilled Wan2.2 I2V model directory is incomplete. "
            f"Expected model files under {model_path}. Missing: {missing_text}. "
            "Use --ckpt_dir to point to the release model directory, or create "
            "a local ./LightX2VModel symlink."
        )
    if not any((model_path / "high_noise_model").glob("*.safetensors")):
        raise FileNotFoundError(
            f"No high-noise .safetensors checkpoint found under {model_path / 'high_noise_model'}"
        )
    if not any((model_path / "low_noise_model").glob("*.safetensors")):
        raise FileNotFoundError(
            f"No low-noise .safetensors checkpoint found under {model_path / 'low_noise_model'}"
        )
    return model_path


def validate_original_wan_i2v_model_dir(model_dir) -> Path:
    model_path = Path(model_dir)
    required_paths = [
        model_path / "Wan2.1_VAE.pth",
        model_path / "models_t5_umt5-xxl-enc-bf16.pth",
        model_path / "high_noise_model",
        model_path / "high_noise_model" / "config.json",
        model_path / "low_noise_model",
        model_path / "low_noise_model" / "config.json",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Original Wan2.2 I2V checkpoint directory is incomplete. "
            f"Expected full Wan2.2 model files under {model_path}. Missing: {missing_text}. "
            "Use --ckpt_dir to point to ./Wan2.2-I2V-A14B, or remove --use_original_wan "
            "to use the LightX2V step-distilled backend."
        )
    if not any((model_path / "high_noise_model").glob("*.safetensors")):
        raise FileNotFoundError(
            f"No high-noise .safetensors checkpoint found under {model_path / 'high_noise_model'}"
        )
    if not any((model_path / "low_noise_model").glob("*.safetensors")):
        raise FileNotFoundError(
            f"No low-noise .safetensors checkpoint found under {model_path / 'low_noise_model'}"
        )
    return model_path


def prompt_video_max_area(size: str) -> int:
    from wan.configs import MAX_AREA_CONFIGS

    # Both Wan and LightX2V derive output dimensions from max_area and input
    # aspect ratio. A tiny bump avoids floating-point floor artifacts that can
    # turn supported sizes such as 832x480 into 832x464.
    return int(MAX_AREA_CONFIGS[size]) + 1


def _generate_original_wan_video(wan_i2v, image: Image.Image, prompt: str, args):
    from contextlib import contextmanager

    import torch.distributed as dist

    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    guide_scale = args.sample_guide_scale
    if isinstance(guide_scale, float):
        guide_scale = (guide_scale, guide_scale)

    img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(wan_i2v.device)
    frame_num = args.frame_num
    height, width = img.shape[1:]
    aspect_ratio = height / width
    max_area = prompt_video_max_area(args.size)
    lat_h = round(
        np.sqrt(max_area * aspect_ratio)
        // wan_i2v.vae_stride[1]
        // wan_i2v.patch_size[1]
        * wan_i2v.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio)
        // wan_i2v.vae_stride[2]
        // wan_i2v.patch_size[2]
        * wan_i2v.patch_size[2]
    )
    height = lat_h * wan_i2v.vae_stride[1]
    width = lat_w * wan_i2v.vae_stride[2]
    max_seq_len = ((frame_num - 1) // wan_i2v.vae_stride[0] + 1) * lat_h * lat_w
    max_seq_len //= wan_i2v.patch_size[1] * wan_i2v.patch_size[2]
    max_seq_len = int(math.ceil(max_seq_len / wan_i2v.sp_size)) * wan_i2v.sp_size

    seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=wan_i2v.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        (frame_num - 1) // wan_i2v.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=wan_i2v.device,
    )

    mask = torch.ones(1, frame_num, lat_h, lat_w, device=wan_i2v.device)
    mask[:, 1:] = 0
    mask = torch.concat(
        [torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]],
        dim=1,
    )
    mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
    mask = mask.transpose(1, 2)[0]

    n_prompt = wan_i2v.sample_neg_prompt
    if not wan_i2v.t5_cpu:
        wan_i2v.text_encoder.model.to(wan_i2v.device)
        context = wan_i2v.text_encoder([prompt], wan_i2v.device)
        context_null = wan_i2v.text_encoder([n_prompt], wan_i2v.device)
        if args.offload_model:
            wan_i2v.text_encoder.model.cpu()
    else:
        context = wan_i2v.text_encoder([prompt], torch.device("cpu"))
        context_null = wan_i2v.text_encoder([n_prompt], torch.device("cpu"))
        context = [tensor.to(wan_i2v.device) for tensor in context]
        context_null = [tensor.to(wan_i2v.device) for tensor in context_null]

    y = wan_i2v.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(
                        img[None].cpu(),
                        size=(height, width),
                        mode="bicubic",
                    ).transpose(0, 1),
                    torch.zeros(3, frame_num - 1, height, width),
                ],
                dim=1,
            ).to(wan_i2v.device)
        ]
    )[0]
    y = torch.concat([mask, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low_noise = getattr(wan_i2v.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high_noise = getattr(wan_i2v.high_noise_model, "no_sync", noop_no_sync)

    with (
        torch.amp.autocast("cuda", dtype=wan_i2v.param_dtype),
        torch.no_grad(),
        no_sync_low_noise(),
        no_sync_high_noise(),
    ):
        boundary = wan_i2v.boundary * wan_i2v.num_train_timesteps
        if args.sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=wan_i2v.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                args.sample_steps,
                device=wan_i2v.device,
                shift=args.sample_shift,
            )
            timesteps = sample_scheduler.timesteps
        elif args.sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=wan_i2v.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(args.sample_steps, args.sample_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=wan_i2v.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {args.sample_solver}")

        latent = noise
        arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y]}
        arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y]}

        if args.offload_model:
            torch.cuda.empty_cache()

        for timestep in timesteps:
            latent_model_input = [latent.to(wan_i2v.device)]
            timestep_tensor = torch.stack([timestep]).to(wan_i2v.device)
            model = wan_i2v._prepare_model_for_timestep(
                timestep,
                boundary,
                args.offload_model,
            )
            sample_guide_scale = guide_scale[1] if timestep.item() >= boundary else guide_scale[0]
            noise_pred_cond = model(latent_model_input, t=timestep_tensor, **arg_c)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = model(latent_model_input, t=timestep_tensor, **arg_null)[0]
            if args.offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
            latent = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                timestep,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0].squeeze(0)
            del latent_model_input, timestep_tensor

        x0 = [latent]
        if args.offload_model:
            wan_i2v.low_noise_model.cpu()
            wan_i2v.high_noise_model.cpu()
            torch.cuda.empty_cache()
        videos = wan_i2v.vae.decode(x0)

    del noise, latent, x0, sample_scheduler
    if args.offload_model:
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return videos[0]


def generate_prompt_video(video_pipeline, cfg, image_path: Path, prompt: str, save_path: Path, args) -> None:
    from wan.utils.utils import save_video

    ensure_dir(save_path.parent)
    image = Image.open(image_path).convert("RGB")
    seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    with torch.no_grad():
        if args.use_original_wan:
            video = _generate_original_wan_video(video_pipeline, image, prompt, args)
        else:
            video = video_pipeline.generate(
                prompt,
                image,
                max_area=prompt_video_max_area(args.size),
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model,
            )
    save_video(
        tensor=video[None],
        save_file=str(save_path),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    del video
    torch.cuda.empty_cache()
