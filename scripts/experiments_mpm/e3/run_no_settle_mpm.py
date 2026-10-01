"""E2 step 2: drive the cat/cushion MPM contact sim with the CHORD-trained cat motion.

Adapted from MPMAvatar exp16_chord_motion.py; only default settle delay and output routing change.

Replaces the rigid vertical press of exp145's cases with the per-frame deformed
cat mesh exported by CHORDCode's scripts/experiments_mpm/e2/export_chord_cat_motion.py.
All physics settings match exp145 case 05_dx_002 (jelly, baseline 13 numerics).
"""
from pathlib import Path
import sys
import argparse
import importlib.util
import inspect
import hashlib
import json
import numpy as np
import torch
import warp as wp
AVATAR_BASELINES = Path('/home/ydu/code/MPMAvatar/scripts/behavior_baselines')
sys.path.insert(0, str(AVATAR_BASELINES))
from exp10_curved_mesh import base

ROOT = Path('/home/ydu/code/CHORDCode')
HELPER = ROOT / 'scripts/experiments_mpm/e1/cushion_volume_particles.py'
spec = importlib.util.spec_from_file_location('cushion_geometry16', HELPER)
geom = importlib.util.module_from_spec(spec); sys.modules[spec.name] = geom; spec.loader.exec_module(geom)


@wp.kernel
def drive_mesh(traj: wp.array(dtype=wp.vec3, ndim=2),
               points: wp.array(dtype=wp.vec3),
               velocities: wp.array(dtype=wp.vec3),
               s_now: wp.float32,
               s_prev: wp.float32,
               fps: wp.float32):
    i = wp.tid()
    n_frames = traj.shape[0]
    s0 = wp.min(wp.max(s_now, 0.0), wp.float32(n_frames - 1))
    s1 = wp.min(wp.max(s_prev, 0.0), wp.float32(n_frames - 1))
    i0 = wp.min(wp.int32(wp.floor(s0)), n_frames - 2)
    f0 = wp.min(wp.max(s0 - wp.float32(i0), 0.0), 1.0)
    j0 = wp.min(wp.int32(wp.floor(s1)), n_frames - 2)
    g0 = wp.min(wp.max(s1 - wp.float32(j0), 0.0), 1.0)
    p_now = traj[i0, i] + (traj[i0 + 1, i] - traj[i0, i]) * f0
    p_prev = traj[j0, i] + (traj[j0 + 1, i] - traj[j0, i]) * g0
    points[i] = p_now
    ds = wp.max(s_now - s_prev, wp.float32(1.0e-9))
    velocities[i] = (p_now - p_prev) * (fps / ds)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-npz', type=Path,
                        default=ROOT / 'trained/cat_with_cushion/outputs_mpmavatar_e2/chord_cat_motion/chord_cat_motion.npz')
    parser.add_argument('--scene-dir', type=Path, default=ROOT / 'data/cat_with_cushion')
    parser.add_argument('--tag', default='no_settle')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=100)
    parser.add_argument('--domain', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=1.0 / 96000)
    parser.add_argument('--young', type=float, default=200000.0)
    parser.add_argument('--density', type=float, default=1000.0)
    parser.add_argument('--gravity', type=float, default=-9.8)
    parser.add_argument('--pitch', type=float, default=0.02)
    parser.add_argument('--floor', type=float, default=0.18)
    parser.add_argument('--surface', choices=('sticky', 'separate'), default='sticky')
    parser.add_argument('--floor-friction', type=float, default=0.0)
    parser.add_argument('--friction', type=float, default=0.0)
    parser.add_argument('--settle-delay', type=float, default=0.0,
                        help='initial hold at frame-0 pose so the cushion settles under gravity')
    parser.add_argument('--free-tail', type=float, default=5.0 / 12,
                        help='observation period after the last motion frame')
    parser.add_argument('--sample-period', type=float, default=1.0 / 30)
    return parser.parse_args()


def main():
    args = parse_args()
    archive = np.load(args.motion_npz)
    cat_traj_norm = archive['vertices'].astype(np.float32)   # (F, N, 3) CHORD normalized space
    cat_faces = archive['faces'].astype(np.int32)
    fps = int(archive['fps'][0])
    frame_count = int(cat_traj_norm.shape[0])

    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    # Cushion particles with the exact E1 geometry helper, then the exp145 shift rule.
    vol = geom.heightfield_volume_particles(
        geom.load_chord_normalized_mesh(args.scene_dir, 'obj_2.glb'), args.pitch)
    p = vol.positions.copy()
    shift = np.array([.815, .19, .915], dtype=np.float32)
    shift[[0, 2]] -= .5 * (p.min(0) + p.max(0))[[0, 2]]
    shift[1] -= p[:, 1].min()
    p += shift
    cat_traj = cat_traj_norm + shift                       # cat in MPM domain, CHORD orientation
    # The CHORD cat moves laterally, so keep the exp145 cushion/floor placement but
    # translate x/z (never y) until the whole trajectory fits inside the domain.
    margin = 0.19
    flat = cat_traj.reshape(-1, 3)
    extra = np.maximum(margin - flat.min(0), 0.0) + np.minimum(args.domain - margin - flat.max(0), 0.0)
    extra[1] = 0.0
    cat_traj = cat_traj + extra
    p = p + extra
    shift = shift + extra
    verts0 = cat_traj[0]
    if np.any(flat.min(0) + extra < 0.0) or np.any(flat.max(0) + extra >= args.domain):
        raise ValueError('cat trajectory does not fit in the MPM domain')

    n = len(p); grid = args.grid; domain = args.domain; dt = args.dt
    motion_span = frame_count / fps
    duration = args.settle_delay + motion_span + args.free_tail
    steps = round(duration / dt)
    sample_steps = set(np.rint(np.arange(0, duration + 1e-8, args.sample_period) / dt).astype(int).tolist() + [steps])

    # The cat's lowest frame-0 vertices are tail tips beside the cushion, so a
    # frame-0 footprint box selects nothing. Use the union of the cat's XZ
    # reach over all frames instead: top cushion particles under the cat's area.
    lo = cat_traj.reshape(-1, 3)[:, [0, 2]].min(0) - args.pitch
    hi = cat_traj.reshape(-1, 3)[:, [0, 2]].max(0) + args.pitch
    local = (p[:, 1] >= np.quantile(p[:, 1], .8)) & np.all((p[:, [0, 2]] >= lo) & (p[:, [0, 2]] <= hi), axis=1)
    if not local.any():
        raise ValueError('cat reach box selected no top cushion particles')

    wp.init(); device = 'cuda:0'
    positions = torch.as_tensor(p, device=device)
    state = base.MPMStateStruct(); state.init(n, 0, 0, device=device, requires_grad=False)
    empty_m = torch.empty((0, 3, 3), device=device); empty_v = torch.empty((0, 3), device=device)
    state.from_torch(positions, torch.as_tensor(vol.volumes, device=device), empty_m, empty_v, empty_v,
                     np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
                     tensor_velocity=torch.zeros_like(positions), n_grid=grid, grid_lim=domain, device=device, requires_grad=False)
    state.reset_state(0, positions, empty_m, tensor_velocity=torch.zeros_like(positions), tensor_R_inv=empty_v, device=device, requires_grad=False)
    model = base.MPMModelStruct(); model.init(n, device=device, requires_grad=False); model.init_other_params(n_grid=grid, grid_lim=domain, device=device)
    solver = base.MPMWARP(n, 0, 0, n_grid=grid, grid_lim=domain, device=device)
    solver.set_parameters_dict(model, state, {'material': 'jelly', 'g': [0., args.gravity, 0.], 'density': args.density,
                                              'friction_angle': 30., 'grid_v_damping_scale': 1.}, device=device)
    solver.set_E_nu(model, args.young, .2, 0., 0., device=device); solver.prepare_mu_lam(model, state, device=device)
    solver.add_surface_collider([0., args.floor, 0.], [0., 1., 0.], surface=args.surface, friction=args.floor_friction)
    solver.mesh = wp.Mesh(points=wp.from_numpy(verts0, dtype=wp.vec3, device=device),
                          velocities=wp.zeros(len(verts0), dtype=wp.vec3, device=device),
                          indices=wp.from_numpy(cat_faces.reshape(-1), dtype=wp.int32, device=device))
    solver.num_mesh_v = len(verts0); solver.num_mesh_f = len(cat_faces)
    solver.add_mesh_collider(solver.mesh.id, grid, friction=args.friction)
    traj_wp = wp.from_numpy(cat_traj, dtype=wp.vec3, device=device)

    paths = [Path(__file__), HELPER, Path(base.__file__), Path(inspect.getfile(base.MPMWARP)),
             AVATAR_BASELINES / 'exp10_curved_mesh.py', args.motion_npz,
             args.scene_dir / 'obj_0.glb', args.scene_dir / 'obj_2.glb', args.scene_dir / 'scene.glb']
    config = dict(material='jelly', young=args.young, density=args.density, poisson=.2,
                  gravity=[0, args.gravity, 0], grid=grid, domain=domain, dt=dt, pitch=args.pitch,
                  particles=n, particle_volume=float(vol.volumes[0]), floor=args.floor,
                  particle_shift=shift.tolist(), motion_source=str(args.motion_npz),
                  motion=dict(fps=fps, frames=frame_count, settle_delay=args.settle_delay,
                              free_tail=args.free_tail, duration=duration),
                  surface=args.surface, floor_friction=args.floor_friction, friction=args.friction, local_mask_note='top cushion particles inside the union of the cat XZ reach over all frames',
                  case=args.tag, dx=domain / grid,
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    snapshots = []; speeds = []; cat_samples = []; metrics = []; output_steps = []
    failure = None
    for step in range(steps + 1):
        if step in sample_steps:
            x = state.particle_x.numpy(); v = state.particle_v.numpy()
            if not np.isfinite(x).all() or not np.isfinite(v).all():
                failure = dict(reason='nonfinite', time=step * dt); break
            snapshots.append(x.copy()); speeds.append(np.linalg.norm(v, axis=1)); output_steps.append(step)
            s_sample = min(max((step * dt - args.settle_delay) * fps, 0.0), frame_count - 1.0)
            i0 = min(int(np.floor(s_sample)), frame_count - 2)
            f0 = min(max(s_sample - i0, 0.0), 1.0)
            cat_samples.append((1 - f0) * cat_traj[i0] + f0 * cat_traj[i0 + 1])
            boundary = (np.isclose(x, 2 * domain / grid, atol=1e-6) | np.isclose(x, domain - 2 * domain / grid, atol=1e-6)).any(1)
            metrics.append(dict(time=step * dt, mean_y=float(x[:, 1].mean()), height=float(np.ptp(x[:, 1])),
                                max_speed=float(np.linalg.norm(v, axis=1).max()),
                                max_displacement=float(np.linalg.norm(x - p, axis=1).max()),
                                boundary_particle_count=int(boundary.sum()),
                                motion_frame=float(s_sample),
                                local_mean_down=float(-(x - p)[local, 1].mean()),
                                local_max_down=float(-(x - p)[local, 1].min())))
            if len(metrics) % 10 == 1: print(metrics[-1], flush=True)
            if boundary.sum() > n * .1 or np.linalg.norm(v, axis=1).max() > 1.e4:
                failure = dict(reason='boundary_over_10pct_or_speed_over_1e4', time=step * dt); break
        if step < steps:
            t = step * dt
            s_now = (t + dt - args.settle_delay) * fps
            s_prev = (t - args.settle_delay) * fps
            wp.launch(drive_mesh, len(verts0), inputs=[traj_wp, solver.mesh.points, solver.mesh.velocities,
                                                       float(s_now), float(s_prev), float(fps)], device=device)
            solver.p2g2p(model, state, dt, device=device)
    snapshots = np.asarray(snapshots)
    np.savez_compressed(out / 'trajectory.npz', particles=snapshots, speed=np.asarray(speeds),
                        cat_vertices=np.asarray(cat_samples), cat_faces=cat_faces,
                        output_steps=np.asarray(output_steps), local_particle_mask=local)
    summary = dict(dt=dt, pitch=args.pitch, particle_count=n, finite=failure is None or failure["reason"] != "nonfinite",
                   completed=failure is None, failure=failure,
                   peak_local_mean_down=max(m["local_mean_down"] for m in metrics),
                   peak_local_max_down=max(m["local_max_down"] for m in metrics),
                   final_local_mean_down=metrics[-1]["local_mean_down"],
                   max_particle_displacement=float(np.linalg.norm(snapshots - p, axis=2).max()),
                   peak_boundary_particle_count=max(m['boundary_particle_count'] for m in metrics),
                   initial_gap=None, press_travel=None, nominal_intrusion=None,
                   motion=dict(fps=fps, frames=frame_count, settle_delay=args.settle_delay, free_tail=args.free_tail))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    if len(snapshots) > 1:
        render(out)
    print('E3 no-settle chord-motion complete', flush=True)


def render(out):
    """Paired-view render matching exp145_render, but with per-frame cat vertices."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio

    def section(ax, vertices, faces, z):
        lines = []
        for tri in vertices[faces]:
            hits = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                da, db = tri[a, 2] - z, tri[b, 2] - z
                if da * db <= 0 and da != db:
                    hits.append(tri[a, :2] - da / (db - da) * (tri[b, :2] - tri[a, :2]))
            if len(hits) >= 2: lines.append(hits[:2])
        ax.add_collection(LineCollection(lines, colors='crimson', linewidths=.8))

    out = Path(out); a = np.load(out / 'trajectory.npz'); summary = json.loads((out / 'summary.json').read_text())
    p = a['particles']; speeds = a['speed']; verts_seq = a['cat_vertices']; faces = a['cat_faces']
    times = a['output_steps'] * summary['dt']; fps = (len(times) - 1) / (times[-1] - times[0])
    z = float(np.median(p[0, :, 2])); floor = float(p[0, :, 1].min() - .5 * summary['pitch'])
    vmax = max(float(np.percentile(speeds, 98)), 1e-8)
    both = np.concatenate([p.reshape(-1, 3), verts_seq.reshape(-1, 3)], axis=0)
    low = both.min(0) - .04; high = both.max(0) + .04
    frames = []
    for f in range(len(p)):
        m = verts_seq[f]
        fig = plt.figure(figsize=(12, 5.2), dpi=120)
        ax = fig.add_subplot(121, projection='3d')
        dots = ax.scatter(p[f, :, 0], p[f, :, 2], p[f, :, 1], c=speeds[f], s=5, cmap='turbo', vmin=0, vmax=vmax)
        ax.add_collection3d(Poly3DCollection(m[faces][:, :, [0, 2, 1]], alpha=.13, facecolor='gray', edgecolor='none'))
        ax.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), zlim=(low[1], high[1]), xlabel='x', ylabel='z', zlabel='y')
        ax.set_box_aspect((high - low)[[0, 2, 1]]); ax.view_init(elev=20, azim=42)
        ax = fig.add_subplot(122); mask = np.abs(p[f, :, 2] - z) <= .015
        ax.scatter(p[f, mask, 0], p[f, mask, 1], c=speeds[f, mask], s=8, cmap='turbo', vmin=0, vmax=vmax)
        section(ax, m, faces, z); ax.axhline(floor, color='black', lw=1)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='x', ylabel='y',
               title=f'Central section: z={z:.3f} +/- 0.015')
        ax.set_aspect('equal'); fig.colorbar(dots, ax=fig.axes, fraction=.025, pad=.05, label='particle speed')
        fig.suptitle(f'E3 no-settle chord-motion: {out.name} | t={times[f]:.3f}s')
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(out / 'mpm_process.mp4', frames, fps=fps, codec='libx264', quality=8)
    imageio.mimsave(out / 'mpm_process.gif', frames, duration=1000 / fps, loop=0)
    press = int(np.argmax([m['local_mean_down'] for m in json.loads((out / 'metrics.json').read_text())]))
    imageio.imwrite(out / 'press_frame.png', frames[min(press, len(frames) - 1)])
    (out / 'README.md').write_text(
        "# " + out.name + "\n\nCHORD-trained cat motion drives the mesh collider (E2 step 2). "
        "Zero settle delay; Physics settings match exp145 case 05_dx_002; the vertical press is replaced by the "
        "per-frame deformed cat mesh from CHORDCode e2 export.\n", encoding='utf-8')
    print('E3 visualization complete:', out, flush=True)


if __name__ == '__main__':
    main()
