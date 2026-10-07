# Five interaction demos: pure CHORD with TI2V supervision

The five `*_with_*_izar_v100.sh` scripts for the new assets use
`--task ti2v-5B` and `/scratch/izar/ydu/checkpoints/Wan2.2-TI2V-5B`.
Each reads its existing `data/<scene>/prompt.txt` and requires all three final
`iteration_50000/point_cloud.ply` preprocessing outputs.

The default full run uses 3,000 iterations, 41 frames, 832 × 480 pixels,
batch size 1, a 48-hour limit (normal QoS allows 72 hours), and the existing CHORD regularization/control-point settings.
No MPM solver or physical trajectory supervision is enabled. Compared with the
previous I2V template, TI2V uses the full Wan2.2 VAE, CPU text encoding, FP16 DiT
weights on V100, and CPU-offloaded split SDS/render backward. I2V-specific
Tiny VAE and MMGP flags are removed. Render backward uses seven-frame chunks.

TI2V's SDS setup now honors the requested spatial area rather than forcing
704 × 1280, and provides the latent-shape and split-target interfaces used by
CHORD. The original TI2V SDS target equation and first-frame exclusion remain.

Example debug submission (three real optimizer steps, separate output):

```bash
mkdir -p logs
sbatch --qos=debug --time=00:15:00 \
  --job-name=broom_ti2v_debug --output=logs/broom_ti2v_debug-%j.out \
  --export=ALL,ITERATIONS=3,SAVE_INTERVAL=1,EX_NAME=broom_with_trashbag_ti2v_debug \
  scripts/slurm/broom_with_trashbag_izar_v100.sh
```

After successful debug verification:

```bash
sbatch scripts/slurm/broom_with_trashbag_izar_v100.sh
```

Debug output must be separate from the full run; full training starts from the
static Gaussians, not the three-step debug deformation. A smoke test confirms
execution and gradients, not motion quality or the later additional-control-point
stage at iteration 300. Check Slurm completion, finite checkpoint tensors and
actual motion-parameter updates before accepting a debug run.

Official model source: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
