# Per-case pipeline entry point

```bash
python train.py --stage 1 --ca_with_dog --task i2v-A14B --size '416*240' --ckpt_dir ./ckpts/Wan2.2-I2V-A14B --prompt 'The man bends down and rubs the dog head.'
python train.py --stage 1 2 --cat_with_cushion
python train.py --stage 3 --ca_with_dog
```

Case flags are discovered from directories under `data/`; `--case NAME` is also
supported. Default stage is 1. Original CHORD training options are forwarded to
`pipeline/stages/chord.py`. Case selection supplies mesh/model paths and, when
available, `prompt.txt`; it does not apply historical example-script presets.
Explicit model paths and prompts override these defaults. Existing submission
scripts selecting cases through `--mesh_source_path data/<case>` still work.
Use the repository root as working directory for legacy relative paths.

Stage 1 retains the original optimization and output formats, with an added
final-iteration save condition. Generic active mesh export is not integrated yet.
Stage 2 now supports the cat/cushion case using the existing export and zero-settle
MPM experiment. It invokes configurable Python environments (both default to the invoking interpreter),
and uses the local `physics/warp_mpm` backend. It produces
`active_motion/` and `mpm/` under a new output directory; passive 4D mesh is not
implemented. Physics defaults and historical timing conventions are unchanged.

```bash
sbatch scripts/slurm_sss/cat_with_cushion_stage1_debug_izar.sh
sbatch scripts/slurm_sss/cat_with_cushion_stage2_debug_izar.sh
python train.py --stage 2 --cat_with_cushion \
  --chord-python /scratch/izar/ydu/.conda/envs/chord0/bin/python \
  --mpm-python /scratch/izar/ydu/.conda/envs/chord0/bin/python \
  --stage1-experiment cat_with_cushion_izar_v100 --stage1-checkpoint 3000 \
  --mpm-output-dir trained/cat_with_cushion/stage2_manual
```

With `--stage 1 2`, Stage 2 takes the Stage 1 model path, experiment name and
final iteration from the forwarded CHORD options. This initial case-specific
integration requires 41 frames and no early stopping. The output directory must
be new; failed or partial runs are retained, not overwritten. Stage 3 remains
unimplemented and is rejected before any training starts.
