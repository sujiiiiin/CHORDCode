#!/usr/bin/env python3
"""Audit fixed-material-point refinement interfaces; never train or rerun MPM."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments_mpm.e1.cushion_volume_particles import (
    heightfield_volume_particles, load_chord_normalized_mesh,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(a):
    a = np.asarray(a)
    return dict(min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def interpolate(times, values, query):
    """Linear interpolation only inside saved time coverage; no extrapolation."""
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError('Query outside trajectory time range')
    right = np.clip(np.searchsorted(times, query, side='right'), 1, len(times) - 1)
    left = right - 1
    alpha = (query - times[left]) / (times[right] - times[left])
    shape = (len(query),) + (1,) * (values.ndim - 1)
    return (1 - alpha.reshape(shape)) * values[left] + alpha.reshape(shape) * values[right], left, right, alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpm-dir', type=Path, default=Path('/home/ydu/code/MPMAvatar/output/mpm_visualization/16_chord_motion/chord_motion_fps30'))
    parser.add_argument('--motion-dir', type=Path, default=ROOT / 'trained/cat_with_cushion/outputs_mpmavatar_e2/chord_cat_motion')
    parser.add_argument('--scene-dir', type=Path, default=ROOT / 'data/cat_with_cushion')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'trained/cat_with_cushion')
    parser.add_argument('--experiment', default='cat_with_cushion_izar_v100')
    parser.add_argument('--checkpoint', type=int, default=3000)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--geometry-only', action='store_true', help='Skip CUDA checkpoint queries; not a full interface pass')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(checks={}, warnings=[], geometry_only=args.geometry_only)

    def check(name, passed, details):
        report['checks'][name] = dict(passed=bool(passed), details=details)

    try:
        config = json.loads((args.mpm_dir / 'config.json').read_text())
        summary = json.loads((args.mpm_dir / 'summary.json').read_text())
        motion_stats = json.loads((args.motion_dir / 'motion_stats.json').read_text())
        motion_path = args.motion_dir / 'chord_cat_motion.npz'
        motion = np.load(motion_path, allow_pickle=False)
        sim = np.load(args.mpm_dir / 'trajectory.npz', allow_pickle=False)
        particles = sim['particles']; cat = motion['vertices']
        shift = np.asarray(config['particle_shift'], dtype=np.float64)
        times = sim['output_steps'].astype(np.float64) * config['dt']
        fps = float(motion['fps'][0]); frames = len(cat)
        delay = config['motion']['settle_delay']
        frame_times = delay + np.arange(frames) / fps
        check('completed_finite_trajectory', summary['completed'] and summary['finite'] and
              np.isfinite(particles).all() and np.isfinite(sim['cat_vertices']).all(), dict(shape=list(particles.shape)))
        check('time_metadata', fps == config['motion']['fps'] and frames == config['motion']['frames'] and
              len(times) == len(particles) and times[0] == 0 and np.all(np.diff(times) > 0),
              dict(fps=fps, frames=frames, dt=config['dt']))
        source_checks = {}
        for path, expected in config['source_sha256'].items():
            p = Path(path)
            source_checks[path] = 'match' if p.exists() and sha256(p) == expected else 'missing_or_changed'
        # A changed runner is recorded; asset and motion provenance are mandatory.
        required = [config['motion_source']] + [str(args.scene_dir / n) for n in ('scene.glb', 'obj_0.glb', 'obj_2.glb')]
        check('input_provenance', all(source_checks.get(p) == 'match' for p in required) and
              sha256(motion_path) == config['source_sha256'][config['motion_source']], source_checks)
        if any(v != 'match' for v in source_checks.values()):
            report['warnings'].append('Some recorded source files changed or are missing; see provenance details.')
        scene_mesh = __import__('trimesh').load(args.scene_dir / 'scene.glb', force='mesh', process=False)
        center = scene_mesh.bounds.mean(0); scale = 1.2 / np.ptp(scene_mesh.bounds, axis=0).max()
        check('scene_normalization', np.allclose(center, motion['center'], atol=1e-6, rtol=0) and
              np.allclose(scale, motion['scale'], atol=1e-8, rtol=0),
              dict(center=center.tolist(), scale=float(scale), shift=shift.tolist(),
                   mapping='x_mpm = (x_asset - center) * scale + particle_shift; no rotation or extra scale'))
        vol = heightfield_volume_particles(load_chord_normalized_mesh(args.scene_dir), config['pitch'])
        reference = particles[0].astype(np.float64) - shift
        ordered_error = float(np.max(np.abs(reference - vol.positions))) if reference.shape == vol.positions.shape else float('inf')
        check('ordered_particle_identity', ordered_error < 2e-6, dict(max_abs_error=ordered_error,
              count=len(reference), mapping='row p stays material point p; no nearest-neighbor reassignment'))
        check('cat_topology', np.array_equal(motion['faces'], sim['cat_faces']), dict(faces=len(motion['faces'])))
        # Validate the actual saved collider poses against the exported cat timeline.
        sample_frame = np.clip((times - delay) * fps, 0, frames - 1)
        expected_cat, _, _, _ = interpolate(np.arange(frames, dtype=float), cat, sample_frame)
        cat_error = float(np.max(np.abs(expected_cat + shift - sim['cat_vertices'])))
        check('cat_space_time_replay', cat_error < 3e-6, dict(max_abs_error=cat_error))
        targets, left, right, alpha = interpolate(times, particles.astype(np.float64) - shift, frame_times)
        nearest_time = np.min(np.abs(frame_times[:, None] - times[None]), axis=1)
        report['time_alignment'] = dict(mapping='t_mpm = settle_delay + chord_frame / fps',
            frame_times=frame_times.tolist(), nearest_saved_time_error=stats(nearest_time),
            interpolated_frame_count=int((nearest_time > 1e-8).sum()), left=left.tolist(), right=right.tolist(), alpha=alpha.tolist(),
            cat_last_motion_time=float(frame_times[-1]), saved_end=float(times[-1]),
            note='Target positions are linear approximations between saved MPM states, not exact substep states.')
        report['settling'] = dict(displacement_at_chord_frame0=stats(np.linalg.norm(targets[0] - reference, axis=1)),
                                  note='Keep rest material coordinates; do not silently subtract the settled displacement.')
        if delay > 0:
            report['warnings'].append('MPM settling means CHORD frame 0 targets are generally not the rest state.')
        if np.any(nearest_time > 1e-8):
            report['warnings'].append('Saved MPM frames require temporal interpolation; exact target accuracy needs denser output or replay.')
        bundle = dict(reference_points=reference.astype(np.float32), target_positions=targets.astype(np.float32),
                      chord_frames=np.arange(frames), mpm_times=frame_times, particle_ids=np.arange(len(reference)),
                      particle_shift=shift, source_times=times, interpolation_left=left,
                      interpolation_right=right, interpolation_alpha=alpha)
        report['input_hashes'] = {str(p): sha256(p) for p in (
            args.mpm_dir / 'config.json', args.mpm_dir / 'trajectory.npz', motion_path,
            args.motion_dir / 'motion_stats.json', Path(__file__))}
        if not args.geometry_only:
            import torch
            from scripts.experiments_mpm.e2.export_chord_cat_motion import load_dynamic_model
            from scene.dynamic_gaussian_model import sample_gs_pdf_warp
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required; use the Slurm script or --geometry-only')
            checkpoint_dir = args.model_dir / args.experiment / 'deform' / f'deform_{args.checkpoint}'
            cushion_path = checkpoint_dir / 'obj_2.pth'; cat_path = checkpoint_dir / 'obj_0.pth'
            report['input_hashes'].update({str(p): sha256(p) for p in (cushion_path, cat_path)})
            model = load_dynamic_model(args.model_dir / 'obj_2', cushion_path, frames)
            motion_params = [model.cp_deform.cp_deform]
            if model.c_cp_deform is not None:
                motion_params.append(model.c_cp_deform.cp_deform)
            # Whitelist motion only; no optimizer or parameter step is created.
            for p in (model.cp_radius, model.cp_rotation, model.c_cp_radius, model.c_cp_rotation):
                if p is not None:
                    p.requires_grad_(False)
            for p in motion_params:
                p.requires_grad_(True)
            points = torch.as_tensor(reference, device='cuda', dtype=torch.float32)
            with torch.no_grad():
                predicted = np.stack([model.query_xyz_time(points, t).cpu().numpy() for t in range(frames)])
                layers = {}
                for label, additional in [('base', False), ('additional', True)]:
                    if additional and model.c_cp_deform is None:
                        continue
                    fn = model.cal_nn_weight_additional if additional else model.cal_nn_weight
                    weights, distances, idx = fn(points, detach_node_radius=True)
                    centers = model.c_cp_center if additional else model.cp_center
                    radii = model.c_cp_radius if additional else model.cp_radius
                    rotations = model.c_cp_rotation if additional else model.cp_rotation
                    raw = sample_gs_pdf_warp(points[:, None].expand(-1, model.n_cp_num, -1),
                        centers[idx, :3], model.gaussians.scaling_activation(radii[idx]), rotations[idx])
                    layers[label] = dict(nearest_cp_distance=stats(torch.sqrt(distances[:, 0]).cpu().numpy()),
                        floor_dominated_fraction=float((raw.sum(-1) < model.n_cp_num * 1e-7).float().mean().item()),
                        raw_weight_sum=stats(raw.sum(-1).cpu().numpy()),
                        weight_sum_error=float((weights.sum(-1) - 1).abs().max().item()))
                check('finite_particle_queries', np.isfinite(predicted).all(), dict(shape=list(predicted.shape)))
                report['control_point_coverage'] = layers
                # Check mesh-style point querying agrees with the Gaussian position path.
                gx = model.gaussians.get_xyz
                equivalence = []
                for t in (0, frames // 2, frames - 1):
                    a = model.query_xyz_time(gx, t)
                    b, _ = model.get_xyz_rotation(t)
                    equivalence.append(float((a - b).abs().max().item()))
                check('query_render_position_equivalence', max(equivalence) < 3e-6, dict(max_abs_errors=equivalence))
                cat_model = load_dynamic_model(args.model_dir / 'obj_0', cat_path, frames)
                cat_rest = load_chord_normalized_mesh(args.scene_dir, 'obj_0.glb').vertices
                cat_points = torch.as_tensor(np.asarray(cat_rest), device='cuda', dtype=torch.float32)
                cat_checkpoint_error = max(float(np.max(np.abs(cat_model.query_xyz_time(cat_points,t).cpu().numpy()-cat[t]))) for t in range(frames))
                check('cat_checkpoint_export_equivalence', cat_checkpoint_error < 3e-6, dict(max_abs_error=cat_checkpoint_error))
            bundle['chord_before_positions'] = predicted
            report['baseline'] = dict(per_frame_position_rmse=np.sqrt(np.mean((predicted-targets)**2, axis=(1,2))).tolist(),
                frame0_rest_error=stats(np.linalg.norm(predicted[0]-reference, axis=1)),
                frame0_target_error=stats(np.linalg.norm(predicted[0]-targets[0], axis=1)),
                note='Unoptimized baseline errors, not refinement results.')
            q0 = model.query_xyz_time(points, 0)
            report['frame0_requires_grad'] = q0.requires_grad
            # Probe connection only; no backward through MPM, no optimization.
            t = frames // 2
            pred = model.query_xyz_time(points, t)
            target = torch.as_tensor(targets[t], device='cuda', dtype=torch.float32)
            loss = ((pred - target)**2).mean()
            grads = torch.autograd.grad(loss, motion_params, allow_unused=True)
            norms = [None if g is None else float(g.norm().item()) for g in grads]
            check('motion_gradient_connection', any(n is not None and n > 0 for n in norms) and
                all(g is None or torch.isfinite(g).all().item() for g in grads), dict(frame=t, mse=float(loss.item()), gradient_norms=norms))
            if not q0.requires_grad and np.max(np.linalg.norm(predicted[0]-targets[0],axis=1)) > 1e-5:
                report['warnings'].append('Frame 0 is detached and mismatches settled target: a full-frame absolute loss has an untrainable first-frame residual.')
            if any(v['floor_dominated_fraction'] > 0 for v in layers.values()):
                report['warnings'].append('Some volume points rely on the 1e-7 weight floor; normalized weights alone do not prove geometric coverage.')
        np.savez_compressed(args.output_dir / 'aligned_interface.npz', **bundle)
        report['status'] = ('PASS_WITH_CAVEATS' if report['warnings'] else 'PASS') if all(c['passed'] for c in report['checks'].values()) else 'FAIL'
        if args.geometry_only and report['status'] != 'FAIL':
            report['status'] = 'GEOMETRY_ONLY_NOT_FULL_PASS'
        report['ready_for_minimal_position_fit'] = (not args.geometry_only and report['status'] != 'FAIL'
            and not np.any(nearest_time > 1e-8)
            and max(report['baseline']['frame0_target_error'].values()) < 1e-5)
    except Exception as exc:
        report['status'] = 'ERROR'
        report['error'] = repr(exc)
        raise
    finally:
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
