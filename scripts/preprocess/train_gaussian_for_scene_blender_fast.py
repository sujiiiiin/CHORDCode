#!/usr/bin/env python3
"""Fast scene training: bake meshes once with Blender, then train with direct rasterized baked textures."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, OptimizationParams
from train_gaussian_per_obj import training as base_training
from utils.general_utils import safe_state


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _iter_source_tree(source_root: Path):
    for cur_root, dirs, files in os.walk(source_root):
        dirs.sort()
        files.sort()
        cur_root_path = Path(cur_root)
        yield cur_root_path, files


def _run_bake(
    python_exe: str,
    bake_script: Path,
    mesh_source_path: Path,
    mesh_file: Path,
    output_mesh: Path,
    args,
) -> None:
    cmd = [
        python_exe,
        str(bake_script),
        "--mesh_source_path",
        str(mesh_source_path),
        "--mesh_file",
        str(mesh_file),
        "--blender_env_exr",
        str(args.blender_env_exr),
        "--output_mesh",
        str(output_mesh),
        "--samples",
        str(args.bake_samples),
        "--margin",
        str(args.bake_margin),
        "--world_strength",
        str(args.bake_world_strength),
        "--device",
        str(args.bake_device),
        "--alpha_threshold",
        str(args.bake_alpha_threshold),
    ]
    if args.bake_texture_size > 0:
        cmd.extend(["--texture_size", str(args.bake_texture_size)])

    print(f"[Bake] {mesh_file} -> {output_mesh}")
    subprocess.run(cmd, check=True)


def prepare_baked_mesh_source(args) -> str:
    source_root = Path(args.mesh_source_path).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"mesh_source_path does not exist: {source_root}")

    bake_only_name = str(getattr(args, "bake_only_cur_train", "")).strip()

    baked_root = (
        Path(args.baked_mesh_root).expanduser().resolve()
        if args.baked_mesh_root
        else (Path(args.model_path).resolve() / "_baked_mesh_cache")
    )

    # Avoid recursive os.walk when users put baked cache under source path.
    if source_root == baked_root or source_root in baked_root.parents:
        raise ValueError(
            f"baked mesh root must be outside mesh_source_path: {baked_root} vs {source_root}"
        )

    bake_script = Path(args.bake_script_path).resolve()
    if not bake_script.exists():
        raise FileNotFoundError(f"bake script not found: {bake_script}")

    baked_root.mkdir(parents=True, exist_ok=True)

    baked_count = 0
    linked_count = 0
    for cur_root, files in _iter_source_tree(source_root):
        rel_dir = cur_root.relative_to(source_root)
        out_dir = baked_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src_file = cur_root / filename
            dst_file = out_dir / filename

            if src_file.suffix.lower() == ".glb":
                if bake_only_name and src_file.stem != bake_only_name:
                    _safe_link_or_copy(src_file, dst_file)
                    linked_count += 1
                    continue
                if dst_file.exists() and not args.force_rebake:
                    continue
                if dst_file.exists() and args.force_rebake:
                    dst_file.unlink()
                _run_bake(
                    python_exe=sys.executable,
                    bake_script=bake_script,
                    mesh_source_path=source_root,
                    mesh_file=src_file,
                    output_mesh=dst_file,
                    args=args,
                )
                baked_count += 1
            else:
                _safe_link_or_copy(src_file, dst_file)
                linked_count += 1

    if baked_count == 0:
        print("[Bake] No .glb files were baked (all reused or none found).")
    elif bake_only_name:
        print(f"[Bake] Baked only --cur_train target: {bake_only_name}")
    print(f"[Bake] Finished. baked_glb={baked_count}, linked_non_glb={linked_count}")
    print(f"[Bake] Using baked mesh source: {baked_root}")
    return str(baked_root)


def main() -> None:
    parser = ArgumentParser(description="Training script parameters (fast baked-mesh mode)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--zoom_surface_objs", nargs="+", type=str, default=[])

    parser.add_argument("--blender_env_exr", type=str, required=True)
    parser.add_argument(
        "--bake_script_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "bake_mesh_blender.py"),
    )
    parser.add_argument("--baked_mesh_root", type=str, default="")
    parser.add_argument("--force_rebake", action="store_true")
    parser.add_argument("--bake_texture_size", type=int, default=0)
    parser.add_argument("--bake_samples", type=int, default=512)
    parser.add_argument("--bake_margin", type=int, default=16)
    parser.add_argument("--bake_world_strength", type=float, default=1.0)
    parser.add_argument("--bake_device", type=str, default="AUTO", choices=["AUTO", "CUDA", "CPU"])
    parser.add_argument("--bake_alpha_threshold", type=float, default=0.999)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    baked_source_path = prepare_baked_mesh_source(args)

    dataset = lp.extract(args)
    dataset.mesh_source_path = baked_source_path

    opt = op.extract(args)
    # Force direct rasterization path (no runtime lighting) for GT renders.
    if hasattr(opt, "use_light"):
        opt.use_light = False
    if hasattr(opt, "light_bg"):
        opt.light_bg = False

    base_training(
        dataset,
        opt,
        args.save_iterations,
        args.start_checkpoint,
        args.zoom_surface_objs,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
