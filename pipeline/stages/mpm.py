"""Reuse the existing cat/cushion export and forward MPM experiment."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def validate_mpm_stage(case, args):
    if case != "cat_with_cushion":
        raise NotImplementedError("Stage 2 is implemented only for cat_with_cushion.")
    for executable in (args.chord_python, args.mpm_python):
        if not Path(executable).is_file():
            raise FileNotFoundError(executable)
    if args.mpm_output_dir is None:
        raise ValueError("Stage 2 requires --mpm-output-dir (a new output directory).")
    if args.mpm_output_dir.exists():
        raise FileExistsError(args.mpm_output_dir)


def run_mpm_stage(case, args):
    validate_mpm_stage(case, args)
    out = args.mpm_output_dir.resolve()
    motion_dir = out / "active_motion"
    # The export process runs in the original CHORD environment.
    subprocess.run([
        args.chord_python, str(ROOT / "scripts/evaluation/export_chord_cat_motion.py"),
        "--scene-dir", str(ROOT / "data" / case),
        "--model-dir", str(args.stage1_model_dir),
        "--experiment", args.stage1_experiment,
        "--checkpoint", str(args.stage1_checkpoint),
        "--frame-count", "41", "--fps", "30",
        "--output-dir", str(motion_dir),
    ], cwd=ROOT, check=True)
    # Keep physics defaults identical to the successful zero-settle E3 experiment.
    sim_dir = out / "mpm"
    subprocess.run([
        args.mpm_python, str(ROOT / "physics/cat_cushion_mpm.py"),
        "--motion-npz", str(motion_dir / "chord_cat_motion.npz"),
        "--scene-dir", str(ROOT / "data" / case),
        "--settle-delay", "0", "--output-dir", str(sim_dir),
    ], cwd=ROOT, check=True)
    summary = json.loads((sim_dir / "summary.json").read_text())
    if not summary.get("completed") or not summary.get("finite"):
        raise RuntimeError(f"MPM did not complete successfully: {summary.get('failure')}")
    for name in ("trajectory.npz", "mpm_process.mp4"):
        if not (sim_dir / name).is_file():
            raise FileNotFoundError(sim_dir / name)
    print(f"[Stage 2] Completed particle simulation and video: {sim_dir}", flush=True)

