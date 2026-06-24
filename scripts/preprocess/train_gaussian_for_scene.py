#!/usr/bin/env python3
"""Train all object/background Gaussians for one CHORD scene."""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MANAGED_CHILD_ARGS = {
    "-m",
    "--model_path",
    "--mesh_source_path",
    "--cur_train",
    "--zoom_surface_objs",
    "--elev_r",
    "--near_cam_radius",
    "--zoom_in_train",
    "--light_bg",
    "--static_id",
}

DEFAULT_CHILD_ARGS = [
    "--densify_grad_threshold",
    "0.0003",
    "--cam_radius",
    "2.0",
    "--densification_interval",
    "100",
    "--densify_from_iter",
    "1000",
    "--densify_until_iter",
    "35000",
    "--image_width",
    "1280",
    "--image_height",
    "704",
    "--lambda_depth",
    "1.0",
    "--move_cam_radius",
    "0.1",
]


def _normalize_obj_name(value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("Object names must be non-empty.")
    return value if value.startswith("obj_") else f"obj_{value}"


def _obj_sort_key(obj_name: str) -> tuple[int, str]:
    match = re.fullmatch(r"obj_(\d+)", obj_name)
    if match:
        return int(match.group(1)), obj_name
    return sys.maxsize, obj_name


def _discover_objects(mesh_source_path: str) -> list[str]:
    mesh_root = Path(mesh_source_path)
    if not mesh_root.is_dir():
        raise FileNotFoundError(f"mesh_source_path does not exist or is not a directory: {mesh_root}")

    obj_names = []
    for path in mesh_root.iterdir():
        if path.is_file() and re.fullmatch(r"obj_\d+\.glb", path.name):
            obj_names.append(path.stem)

    if not obj_names:
        raise FileNotFoundError(f"No top-level obj_<id>.glb files found under {mesh_root}")
    return sorted(obj_names, key=_obj_sort_key)


def _select_training_order(discovered: Sequence[str], requested: Iterable[str], static_id: int, skip_static: bool) -> list[str]:
    discovered_set = set(discovered)
    static_name = f"obj_{static_id}"

    if requested:
        selected = [_normalize_obj_name(value) for value in requested]
        missing = [obj_name for obj_name in selected if obj_name not in discovered_set]
        if missing:
            raise FileNotFoundError(
                "Requested object meshes were not found: "
                + ", ".join(missing)
                + f". Available objects: {', '.join(discovered)}"
            )
    else:
        selected = list(discovered)

    dynamic_objs = sorted((obj_name for obj_name in selected if obj_name != static_name), key=_obj_sort_key)
    train_order = dynamic_objs
    if not skip_static and static_name in selected:
        train_order.append(static_name)

    if not train_order:
        raise ValueError("No objects selected for Gaussian training.")
    return train_order


def _reject_managed_passthrough(extra_args: Sequence[str]) -> None:
    rejected = []
    for token in extra_args:
        key = token.split("=", 1)[0]
        if key in MANAGED_CHILD_ARGS:
            rejected.append(token)
    if rejected:
        raise ValueError(
            "These options are managed by train_gaussian_for_scene.py and cannot be forwarded: "
            + ", ".join(rejected)
        )


def _build_child_command(args, obj_name: str, extra_args: Sequence[str]) -> list[str]:
    is_static = obj_name == f"obj_{args.static_id}"
    train_script = Path(args.train_script)
    if not train_script.is_absolute():
        train_script = Path(__file__).resolve().parent / train_script
    if not train_script.exists():
        raise FileNotFoundError(f"Per-object Gaussian training script not found: {train_script}")

    cmd = [
        sys.executable,
        str(train_script),
        "-m",
        args.model_path,
        "--mesh_source_path",
        args.mesh_source_path,
        "--cur_train",
        obj_name,
        "--static_id",
        str(args.static_id),
        *DEFAULT_CHILD_ARGS,
    ]

    if not args.no_light:
        cmd.append("--use_light")

    if is_static:
        cmd.extend(["--elev_r", str(args.static_elev_r)])
        cmd.extend(["--near_cam_radius", str(args.static_near_cam_radius)])
    else:
        cmd.extend(["--zoom_surface_objs", obj_name])
        cmd.extend(["--elev_r", str(args.dynamic_elev_r)])
        cmd.extend(["--near_cam_radius", str(args.dynamic_near_cam_radius)])
        if not args.no_zoom_in_train:
            cmd.append("--zoom_in_train")
        if not args.no_light and not args.no_light_bg:
            cmd.append("--light_bg")

    cmd.extend(extra_args)
    return cmd


def parse_args(argv: Sequence[str]):
    parser = argparse.ArgumentParser(
        description="Train all object/background Gaussians for a CHORD scene."
    )
    parser.add_argument("-m", "--model_path", required=True, help="Output directory for fitted Gaussians")
    parser.add_argument("--mesh_source_path", required=True, help="Scene mesh directory containing obj_<id>.glb files")
    parser.add_argument("--objects", nargs="+", default=[], help="Optional subset of objects to train, e.g. obj_0 obj_2")
    parser.add_argument("--static_id", type=int, default=1, help="Object id treated as the static background")
    parser.add_argument("--skip_static", action="store_true", help="Do not train the static background object")
    parser.add_argument("--dry_run", action="store_true", help="Print per-object commands without running them")
    parser.add_argument(
        "--train_script",
        default="train_gaussian_per_obj.py",
        help="Per-object Gaussian training script to invoke",
    )
    parser.add_argument("--dynamic_elev_r", type=float, default=60.0)
    parser.add_argument("--static_elev_r", type=float, default=10.0)
    parser.add_argument("--dynamic_near_cam_radius", type=float, default=0.3)
    parser.add_argument("--static_near_cam_radius", type=float, default=0.5)
    parser.add_argument("--no_light", action="store_true", help="Do not pass --use_light to per-object training")
    parser.add_argument("--no_zoom_in_train", action="store_true", help="Do not pass --zoom_in_train for dynamic objects")
    parser.add_argument("--no_light_bg", action="store_true", help="Do not pass --light_bg for dynamic objects")
    args, extra_args = parser.parse_known_args(argv)
    try:
        _reject_managed_passthrough(extra_args)
    except ValueError as exc:
        parser.error(str(exc))
    return args, extra_args


def main(argv: Sequence[str]) -> None:
    args, extra_args = parse_args(argv)
    discovered = _discover_objects(args.mesh_source_path)
    train_order = _select_training_order(discovered, args.objects, args.static_id, args.skip_static)

    os.makedirs(args.model_path, exist_ok=True)
    print(f"[Gaussian Scene] Found objects: {', '.join(discovered)}")
    print(f"[Gaussian Scene] Training order: {', '.join(train_order)}")

    for obj_name in train_order:
        cmd = _build_child_command(args, obj_name, extra_args)
        print(f"[Gaussian Scene] {obj_name}: {shlex.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if args.dry_run:
        print("[Gaussian Scene] Dry run complete.")
    else:
        print("[Gaussian Scene] Training complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
