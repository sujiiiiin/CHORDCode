#!/usr/bin/env python3
"""Fixed-cat, position-only refinement; learn cushion CP motion, no SDS/MPM backward."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.experiments_mpm.e2.export_chord_cat_motion import load_dynamic_model
from scripts.experiments_mpm.e1.cushion_volume_particles import load_chord_normalized_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    report = json.loads((args.interface_dir / 'report.json').read_text())
    if not report.get('ready_for_minimal_position_fit'):
        raise ValueError('Interface check did not authorize exact-aligned minimal position fit')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.interface_dir / 'aligned_interface.npz')
    points = torch.as_tensor(data['reference_points'], device='cuda')
    targets = torch.as_tensor(data['target_positions'], device='cuda')
    frames, n, _ = targets.shape
    model_dir = ROOT / 'trained/cat_with_cushion'
    checkpoint = model_dir / 'cat_with_cushion_izar_v100/deform/deform_3000/obj_2.pth'
    model = load_dynamic_model(model_dir / 'obj_2', checkpoint, frames)
    params = [model.cp_deform.cp_deform]
    if model.c_cp_deform is not None:
        params.append(model.c_cp_deform.cp_deform)
    for p in (model.cp_radius, model.cp_rotation, model.c_cp_radius, model.c_cp_rotation):
        if p is not None:
            p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    initial_params = [p.detach().clone() for p in params]
    ids = rng.permutation(n)
    valid_ids = ids[:n // 5]
    train_ids = ids[n // 5:]
    np.savez(args.output_dir / 'split.npz', train=train_ids, validation=valid_ids)
    mesh = load_chord_normalized_mesh(ROOT / 'data/cat_with_cushion')
    mesh_points = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device='cuda')
    @torch.no_grad()
    def query_all(x):
        return torch.stack([model.query_xyz_time(x, t) for t in range(frames)])
    @torch.no_grad()
    def evaluate():
        pred = query_all(points)
        delta = pred[1:] - targets[1:]
        # Euclidean point RMSE, not per-coordinate RMSE.
        train_rmse = float(delta[:, train_ids].square().sum(-1).mean().sqrt().item())
        valid_rmse = float(delta[:, valid_ids].square().sum(-1).mean().sqrt().item())
        return pred, train_rmse, valid_rmse
    before, initial_train, initial_valid = evaluate()
    before_mesh = query_all(mesh_points).cpu().numpy()
    config = dict(steps=args.steps, lr=args.lr, seed=args.seed, optimized='cushion base/additional CP motion only',
                  fixed='cat motion, Gaussian attributes, CP centers/radii/rotations, MPM targets',
                  loss='mean squared coordinate distance; frames 1..40', no_sds=True,
                  particles_per_step=min(1024,len(train_ids)), frames_per_step=4,
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  interface_dir=str(args.interface_dir.resolve()),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output_dir / 'config.json').write_text(json.dumps(config,indent=2)+'\n')
    optimizer = torch.optim.Adam(params, lr=args.lr)
    history = [dict(step=0, train_rmse=initial_train, validation_rmse=initial_valid)]
    best = initial_valid
    best_step = 0
    best_params = initial_params
    for step in range(1,args.steps+1):
        chosen = rng.choice(train_ids, size=min(1024,len(train_ids)), replace=False)
        chosen_frames = rng.choice(np.arange(1,frames),size=4,replace=False)
        x = points[chosen]
        optimizer.zero_grad(set_to_none=True)
        loss = sum((model.query_xyz_time(x,int(t))-targets[t,chosen]).square().mean() for t in chosen_frames)/len(chosen_frames)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            _, train_error, valid_error = evaluate()
            row = dict(step=step,train_rmse=train_error,validation_rmse=valid_error)
            history.append(row);print(row,flush=True)
            if valid_error < best:
                best=valid_error;best_step=step
                best_params=[p.detach().clone() for p in params]
    with torch.no_grad():
        for p,b in zip(params,best_params):
            p.copy_(b)
    after, final_train, final_valid = evaluate()
    after_mesh=query_all(mesh_points).cpu().numpy()
    model.save_pth(str(args.output_dir/'obj_2_refined.pth'))
    # Verify the standard saved checkpoint reproduces the result.
    reload_model=load_dynamic_model(model_dir/'obj_2',args.output_dir/'obj_2_refined.pth',frames)
    with torch.no_grad():
        reload_error=max(float((reload_model.query_xyz_time(points,t)-after[t]).abs().max().item()) for t in (0,20,40))
    summary=dict(completed=True,best_step=best_step,initial_train_rmse=initial_train,final_train_rmse=final_train,
                 initial_validation_rmse=initial_valid,final_validation_rmse=final_valid,
                 validation_reduction=1-final_valid/initial_valid,
                 frame0_change_max=float((after[0]-before[0]).abs().max().item()),
                 checkpoint_reload_max_error=reload_error,
                 finite=bool(torch.isfinite(after).all().item()),
                 note='Held-out material points, same 40 times; no held-out-time or physical-accuracy claim.')
    np.savez_compressed(args.output_dir/'comparison.npz',reference=points.cpu().numpy(),
                        target=targets.cpu().numpy(),before=before.cpu().numpy(),after=after.cpu().numpy(),
                        mesh_before=before_mesh,mesh_after=after_mesh,mesh_faces=np.asarray(mesh.faces),
                        times=data['mpm_times'])
    (args.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    render(args.output_dir,model_dir)
    print(json.dumps(summary,indent=2),flush=True)


def render(out,model_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio.v2 as imageio
    a=np.load(out/'comparison.npz')
    cat=np.load(model_dir/'outputs_mpmavatar_e2/chord_cat_motion/chord_cat_motion.npz')
    seqs=[a['target'],a['before'],a['after']]
    all_points=np.concatenate([s.reshape(-1,3) for s in seqs]+[cat['vertices'].reshape(-1,3)],axis=0)
    lo=all_points.min(0)-.03;hi=all_points.max(0)+.03
    # Fixed bounds, same cat, same material points in all columns.
    snapshots=[]
    with imageio.get_writer(out/'position_comparison.mp4',fps=30,codec='libx264',quality=8,macro_block_size=2) as writer:
        for t in range(len(seqs[0])):
            fig=plt.figure(figsize=(15,5),dpi=100)
            for j,(seq,title) in enumerate(zip(seqs,['MPM target','CHORD before','CHORD refined'])):
                ax=fig.add_subplot(1,3,j+1,projection='3d')
                pts=seq[t][:,[0,2,1]]
                ax.scatter(*pts.T,s=1.5,c='#278e91',alpha=.6)
                m=Poly3DCollection(cat['vertices'][t][cat['faces']][:,:,[0,2,1]],alpha=.12,linewidths=0)
                m.set_facecolor('#cf6646');ax.add_collection3d(m)
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2],hi[2]),zlim=(lo[1],hi[1]),title=title)
                ax.set_box_aspect((hi-lo)[[0,2,1]]);ax.view_init(elev=20,azim=42)
                ax.set_axis_off()
            fig.suptitle(f'Fixed cat | frame {t:02d} | particle geometry (not Gaussian rendering)')
            fig.tight_layout();fig.canvas.draw();img=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            writer.append_data(img)
            if t in (0,10,20,30,40):snapshots.append(img)
            plt.close(fig)
    imageio.imwrite(out/'contact_sheet.png',np.concatenate(snapshots,axis=0))
    history=json.loads((out/'history.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4))
    for key in ('train_rmse','validation_rmse'):
        ax.semilogy([r['step'] for r in history],[r[key] for r in history],label=key)
    ax.set(xlabel='Optimization step',ylabel='Euclidean position RMSE (normalized coordinates)');ax.legend();fig.tight_layout()
    fig.savefig(out/'fit_curve.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
