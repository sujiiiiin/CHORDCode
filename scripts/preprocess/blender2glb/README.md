# Blender asset preparation

These tools convert source assets and assemble CHORD input scenes. Gaussian
fitting remains in `scripts/preprocess/train_gaussian_for_scene.py`.
Run commands from the repository root.

```bash
conda env create -p <environment-path> -f scripts/preprocess/blender2glb/environment-mesh.yml
conda activate <environment-path>
python scripts/preprocess/blender2glb/export_blend_assets.py --resume
python scripts/preprocess/blender2glb/build_demo_scenes.py
python scripts/preprocess/blender2glb/check_demo_scenes.py
```

- `export_blend_assets.py`: extract `mydata/3Dassets/*.blend.zip`, export textured
  GLB and OBJ, and write per-asset reports and a manifest.
- `build_demo_scenes.py`: assemble the five approved asset pairs under `data/`,
  including object splits, floor, prompts and static previews. Use `--scene`
  to rebuild a single scene. Rebuilding overwrites that scene's generated files.
- `check_demo_scenes.py`: validate mesh splits, shared world coordinates,
  embedded textures, ground clearance and initial object separation.

The scripts do not launch Gaussian training, Wan supervision or MPM simulation.
Generated assets and scene directories remain excluded by the repository's
existing `.gitignore` rules.
