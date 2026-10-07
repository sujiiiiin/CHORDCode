"""Case selection and stage dispatch; GPU imports are deferred to stage execution."""

import argparse
from pathlib import Path
import sys

from pipeline.stages.mpm import run_mpm_stage, validate_mpm_stage
from pipeline.stages.joint import run_joint_stage

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-case CHORD–MPM pipeline. Stage 2 supports cat/cushion particles; stage 3 is unimplemented.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", nargs="+", type=int, choices=(1, 2, 3), default=[1])
    parser.add_argument("--stage1-model-dir", type=Path, default=ROOT / "trained/cat_with_cushion")
    parser.add_argument("--stage1-experiment", default="cat_with_cushion_izar_v100")
    parser.add_argument("--stage1-checkpoint", type=int, default=3000)
    parser.add_argument("--mpm-output-dir", type=Path)
    parser.add_argument("--chord-python", default=sys.executable)
    parser.add_argument("--mpm-python", default=sys.executable)
    cases = sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=cases)
    for case in cases:
        selection.add_argument(f"--{case}", dest="case", action="store_const", const=case)
    args, remaining = parser.parse_known_args(argv)
    if args.stage != sorted(set(args.stage)):
        parser.error("Stages must be unique and increasing, e.g. --stage 1 2")
    if args.case is None:
        legacy = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        legacy.add_argument("--mesh_source_path")
        paths, _ = legacy.parse_known_args(remaining)
        if paths.mesh_source_path:
            path = Path(paths.mesh_source_path).resolve()
            if path.parent == (ROOT / "data").resolve() and path.name in cases:
                args.case = path.name
        if args.case is None:
            parser.error("Select --case NAME or --NAME; case names come from data/.")
    return args, remaining


def prepare_chord_args(case, argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mesh_source_path")
    parser.add_argument("--model_path", "-m")
    parser.add_argument("--prompt")
    selected, _ = parser.parse_known_args(argv)
    scene_dir = ROOT / "data" / case
    if selected.mesh_source_path and Path(selected.mesh_source_path).resolve() != scene_dir.resolve():
        raise ValueError("--mesh_source_path conflicts with the selected case")
    result = list(argv)
    if selected.mesh_source_path is None:
        result += ["--mesh_source_path", str(scene_dir)]
    if selected.model_path is None:
        result += ["--model_path", str(ROOT / "trained" / case)]
    prompt_file = scene_dir / "prompt.txt"
    if selected.prompt is None and prompt_file.is_file():
        result += ["--prompt", prompt_file.read_text().strip()]
    return result


def run_chord_stage(case, argv):
    options = prepare_chord_args(case, argv)
    from pipeline.stages.chord import run
    run(options)


def main(argv=None):
    args, remaining = parse_args(argv)
    # Preflight all requested stages before starting an expensive GPU optimization.
    if 3 in args.stage:
        run_joint_stage(args.case)
    if 2 in args.stage:
        validate_mpm_stage(args.case, args)
        if 1 in args.stage:
            # Stage 2 must consume this run rather than an unrelated historical checkpoint.
            source = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            source.add_argument("--model_path", "-m", default=str(ROOT / "trained" / args.case))
            source.add_argument("--ex_name", default="default")
            source.add_argument("--iterations", type=int, default=3000)
            source.add_argument("--frame_num", type=int, default=41)
            source.add_argument("--early_stop", action="store_true")
            options, _ = source.parse_known_args(remaining)
            if options.early_stop or options.frame_num != 41:
                raise NotImplementedError("Combined stages currently require 41 frames and no early_stop.")
            args.stage1_model_dir = Path(options.model_path).resolve()
            args.stage1_experiment = options.ex_name
            args.stage1_checkpoint = options.iterations
        else:
            if remaining:
                raise ValueError(f"Unexpected Stage 2 options: {remaining}")
            checkpoint = args.stage1_model_dir / args.stage1_experiment / "deform" / f"deform_{args.stage1_checkpoint}" / "obj_0.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    print(f"[Pipeline] case={args.case}, stages={args.stage}", flush=True)
    if 1 in args.stage:
        run_chord_stage(args.case, remaining)
    if 2 in args.stage:
        run_mpm_stage(args.case, args)
