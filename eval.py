"""Offline visualization for saved CHORD deformation checkpoints.

This entry point intentionally covers only visualizations that can be recovered
from a saved ``deform/deform_<iteration>`` checkpoint. Training-only transient
states (for example, before/after adding control points) are not reconstructible.
"""

import argparse
import copy
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scene import GaussianModel, Scene
from scene.dynamic_gaussian_model import DynamicGaussianModel
from scene.dynamic_scene import DynamicGaussianScene
from utils.general_utils import safe_state
from utils.orbit_cam_utils import OrbitCamera
from utils.render_utils import build_view


_OBJ_DIR_RE = re.compile(r"obj_(\d+)")
_DEFORM_DIR_RE = re.compile(r"deform_(\d+)")


def _parse_size(value):
    match = re.fullmatch(r"(\d+)\*(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("size must use WIDTH*HEIGHT syntax, e.g. 416*240")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def _normalize_obj_names(values):
    names = set()
    for value in values:
        value = str(value).strip()
        if value:
            names.add(value if value.startswith("obj_") else f"obj_{value}")
    return names


def _discover_object_ids(model_path):
    object_ids = []
    for path in model_path.iterdir():
        match = _OBJ_DIR_RE.fullmatch(path.name)
        if path.is_dir() and match:
            object_ids.append(int(match.group(1)))

    if not object_ids:
        raise FileNotFoundError(f"No obj_<id> directories found under {model_path}")

    object_ids.sort()
    expected = list(range(object_ids[-1] + 1))
    if object_ids != expected:
        missing = sorted(set(expected) - set(object_ids))
        raise ValueError(
            "Object directories must be contiguous from obj_0; missing "
            + ", ".join(f"obj_{obj_id}" for obj_id in missing)
        )
    return object_ids


def _discover_checkpoints(experiment_path):
    deform_root = experiment_path / "deform"
    if not deform_root.is_dir():
        raise FileNotFoundError(f"Deformation directory not found: {deform_root}")

    checkpoints = []
    for path in deform_root.iterdir():
        match = _DEFORM_DIR_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append(int(match.group(1)))

    if not checkpoints:
        raise FileNotFoundError(f"No deform_<iteration> checkpoints found under {deform_root}")
    return sorted(checkpoints)


def _resolve_checkpoints(experiment_path, checkpoint, all_checkpoints):
    available = _discover_checkpoints(experiment_path)
    if all_checkpoints:
        return available
    if checkpoint == -1:
        return [available[-1]]
    if checkpoint not in available:
        raise FileNotFoundError(
            f"Checkpoint deform_{checkpoint} is unavailable; found: "
            + ", ".join(str(value) for value in available)
        )
    return [checkpoint]


def _load_dynamic_scene(args, dataset, checkpoint, object_ids):
    dynamic_scene = DynamicGaussianScene(args.frame_num)
    avoided = _normalize_obj_names(args.avoid_objs)
    additional_static = _normalize_obj_names(args.additional_static_objs)
    checkpoint_dir = args.experiment_path / "deform" / f"deform_{checkpoint}"

    loaded_frame_counts = set()
    for obj_id in object_ids:
        obj_name = f"obj_{obj_id}"
        if obj_name in avoided:
            print(f"[Eval] Skipping avoided object {obj_name}")
            continue

        object_dataset = copy.copy(dataset)
        object_dataset.model_path = str(args.model_path / obj_name)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(object_dataset, gaussians, load_iteration=-1)

        is_static = obj_id == args.static_id or obj_name in additional_static
        if is_static:
            dynamic_scene.add_gaussians(gaussians, True, obj_name)
            continue

        checkpoint_path = checkpoint_dir / f"{obj_name}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for dynamic object {obj_name}: {checkpoint_path}. "
                "If this object was static or avoided during training, pass the matching CLI option."
            )

        dynamic_gaussians = DynamicGaussianModel(
            gaussians,
            args.frame_num,
            scene.cameras_extent,
            n_cp_num=args.n_cp_num,
        )
        dynamic_gaussians.load_pth(str(checkpoint_path))
        loaded_frame_counts.add(dynamic_gaussians.total_time)
        dynamic_scene.add_gaussians(dynamic_gaussians, False, obj_name)

    if not dynamic_scene.dynamic_gaussians:
        raise ValueError("No objects remain after applying --avoid_objs")
    if len(loaded_frame_counts) > 1:
        raise ValueError(
            f"Dynamic objects in deform_{checkpoint} have inconsistent frame counts: "
            f"{sorted(loaded_frame_counts)}"
        )
    if loaded_frame_counts:
        saved_frame_num = loaded_frame_counts.pop()
        if saved_frame_num != args.frame_num:
            print(
                f"[Eval] Using checkpoint frame count {saved_frame_num} "
                f"instead of CLI value {args.frame_num}"
            )
        dynamic_scene.total_time = saved_frame_num
    return dynamic_scene


def _has_additional_control_points(dynamic_scene):
    return any(
        not is_static and gaussians.c_cp_deform is not None
        for gaussians, is_static in dynamic_scene.dynamic_gaussians.values()
    )


def _render_checkpoint(args, dataset, checkpoint, object_ids):
    # Importing the video writer initializes Wan modules, so keep it lazy to
    # allow argument inspection (notably --help) on machines without CUDA.
    from utils.eval_utils import save_cp_deform, save_cp_orbit

    print(f"[Eval] Loading checkpoint deform_{checkpoint}")
    dynamic_scene = _load_dynamic_scene(args, dataset, checkpoint, object_ids)

    default_cam = OrbitCamera(
        dataset.image_width,
        dataset.image_height,
        r=args.cam_radius,
        fovy=dataset.fovy,
    )
    reference_elev = (args.elev_l + args.elev_r) * 0.5
    ref_viewmat, ref_opencv_k = build_view(
        dataset,
        default_cam,
        args.ref_cam_radius,
        reference_elev,
        args.ref_azim,
        look_at=None,
    )
    orbit_look_at = np.array([0.0, args.cam_height, 0.0], dtype=np.float32)

    eval_dir = args.output_dir / "eval_videos"
    orbit_dir = args.output_dir / "orbit_videos"
    eval_dir.mkdir(parents=True, exist_ok=True)
    orbit_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        save_cp_deform(
            ref_viewmat,
            ref_opencv_k,
            dataset,
            dynamic_scene,
            str(eval_dir / f"deform_video_{checkpoint}.mp4"),
            scaling_modifier=0.5,
        )
        save_cp_orbit(
            dataset,
            args,
            dynamic_scene,
            str(orbit_dir / f"orbit_video_{checkpoint}.mp4"),
            default_cam,
            look_at=orbit_look_at,
        )
        if _has_additional_control_points(dynamic_scene):
            save_cp_orbit(
                dataset,
                args,
                dynamic_scene,
                str(orbit_dir / f"acp_orbit_video_{checkpoint}.mp4"),
                default_cam,
                additional=True,
            )

    print(f"[Eval] Saved checkpoint deform_{checkpoint} visualizations to {args.output_dir}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate checkpoint-recoverable CHORD training visualizations"
    )
    parser.add_argument(
        "experiment_path",
        type=Path,
        help="Experiment directory containing deform/, e.g. trained/ca_with_dog/ca_with_dog_izar_v100",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Static Gaussian directory containing obj_0, obj_1, ..., e.g. trained/ca_with_dog",
    )
    parser.add_argument("output_dir", type=Path, help="Directory in which to write regenerated MP4 files")

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint iteration; -1 selects the latest (default)",
    )
    checkpoint_group.add_argument(
        "--all_checkpoints",
        action="store_true",
        help="Render every saved deformation checkpoint",
    )

    parser.add_argument(
        "--size",
        type=_parse_size,
        default=(416, 240),
        metavar="WIDTH*HEIGHT",
    )
    parser.add_argument("--fovy", type=float, default=49.1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--frame_num", type=int, default=41)
    parser.add_argument("--n_cp_num", type=int, default=3)
    parser.add_argument("--static_id", type=int, default=1)
    parser.add_argument("--additional_static_objs", nargs="+", default=[])
    parser.add_argument("--avoid_objs", nargs="+", default=[])

    parser.add_argument("--cam_radius", type=float, default=1.8)
    parser.add_argument("--cam_height", type=float, default=0.0)
    parser.add_argument("--elev_l", type=float, default=-10.0)
    parser.add_argument("--elev_r", type=float, default=40.0)
    parser.add_argument("--ref_azim", type=float, default=60.0)
    parser.add_argument("--ref_cam_radius", type=float, default=2.0)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.experiment_path = args.experiment_path.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.experiment_path.is_dir():
        parser.error(f"experiment_path is not a directory: {args.experiment_path}")
    if not args.model_path.is_dir():
        parser.error(f"model_path is not a directory: {args.model_path}")
    if args.checkpoint < -1:
        parser.error("--checkpoint must be -1 or a non-negative iteration")
    if args.frame_num <= 0:
        parser.error("--frame_num must be positive")
    if args.device < 0:
        parser.error("--device must be non-negative")
    return args


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("eval.py requires a CUDA-capable PyTorch environment")
    torch.cuda.set_device(args.device)
    safe_state(args.quiet)

    object_ids = _discover_object_ids(args.model_path)
    if args.static_id not in object_ids:
        raise ValueError(f"--static_id {args.static_id} is not present under {args.model_path}")
    checkpoints = _resolve_checkpoints(
        args.experiment_path,
        args.checkpoint,
        args.all_checkpoints,
    )

    width, height = args.size
    dataset = SimpleNamespace(
        image_width=width,
        image_height=height,
        fovy=args.fovy,
        sh_degree=args.sh_degree,
    )
    for checkpoint in checkpoints:
        _render_checkpoint(args, dataset, checkpoint, object_ids)


if __name__ == "__main__":
    main()
