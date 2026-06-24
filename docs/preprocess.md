# Preparing Custom Data

This guide describes how to prepare a custom scene for CHORD. It covers the mesh layout expected by the optimization code and the object/background 3D Gaussian fitting step.

## Mesh Layout

CHORD expects a mesh directory with one scene mesh and one mesh per object:

```text
data/<scene_name>/
├── scene.glb
├── obj_0.glb
├── obj_1.glb
├── obj_2.glb
└── ...
```

Important conventions:

- `scene.glb` contains the objects whose motion will be generated. It is used to compute scene scale and center for camera sampling.
- `obj_1.glb` is the default static background convention.
- Each `obj_i.glb` must preserve the object's world-space transform.

The image below illustrates the expected object/background split. `obj_1.glb` follows the default static background convention, while the other `obj_i.glb` files can be optimized as dynamic objects.

<p align="center">
  <img src="../assets/scene_example.png" width="50%" height="50%" class="center">
</p>

## Train Scene Gaussians

Train all object and background Gaussians for a scene:

```bash
python scripts/preprocess/train_gaussian_for_scene.py -m trained/<scene_name> --mesh_source_path data/<scene_name>
```

The scene-level script discovers `obj_<id>.glb` files, trains dynamic objects first, and trains the default static background `obj_1` last.

For manual debugging of a single object, use the lower-level `scripts/preprocess/train_gaussian_per_obj.py` script with `--cur_train`.

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/preprocess/train_gaussian_for_scene.py</span></summary>

  #### -m

  Path where the fitted object and background Gaussians are saved, such as `trained/<scene_name>`.

  #### --mesh_source_path

  Path to the scene mesh directory, such as `data/<scene_name>`.

  #### --objects

  Optional subset of objects to train, such as `obj_0 obj_2`. If omitted, all top-level `obj_<id>.glb` files are trained.

  #### --static_id

  Object id used as the static background convention. The release setup uses `1`, corresponding to `obj_1`.

  #### --skip_static

  Train only dynamic objects and skip the static background.

  #### --dry_run

  Print the per-object training commands without running them. This is useful for checking object discovery and training order.

  #### --dynamic_elev_r, --static_elev_r

  Upper elevation bound used for dynamic-object and static-background Gaussian fitting.

  #### --dynamic_near_cam_radius, --static_near_cam_radius

  Near-surface camera radius used for dynamic-object and static-background fitting.

  #### --no_light

  Disable the default simple lighting used during Gaussian fitting.

  #### --no_zoom_in_train

  Do not use zoomed-in training views for dynamic objects.

  #### --no_light_bg

  Do not apply additional lighting to dynamic-object background renders.

</details>

## Preview Static Renders

After fitting the static Gaussians, use `scripts/evaluation/render_scene_static.py` to check the scene composition and prompt framing:

```bash
python scripts/evaluation/render_scene_static.py -m trained/<scene_name> --mesh_source_path data/<scene_name> --checkpoint -1 --image_width 832 --image_height 480 --cam_radius 1.8 --azim_l 0.0 --azim_r 360.0 --elev_l 10.0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/evaluation/render_scene_static.py</span></summary>

  #### -m

  Path to the fitted 3D Gaussian directory for the scene, such as `trained/<scene_name>`.

  #### --mesh_source_path

  Path to the scene mesh directory, such as `data/<scene_name>`.

  #### --obj_num

  Number of object meshes in the scene. The default `-1` auto-detects this from the top-level `obj_i.glb` files in the mesh directory.

  #### --checkpoint

  Gaussian checkpoint iteration to render. Use `-1` to load the latest checkpoint.

  #### --image_width, --image_height

  Output preview resolution in pixels.

  #### --cam_radius

  Camera distance from the scene center.

  #### --azim_l, --azim_r

  Azimuth range, in degrees, for rendering preview views around the scene.

  #### --elev_l

  Camera elevation, in degrees, for the preview orbit. Positive values place the camera above the scene.

</details>
