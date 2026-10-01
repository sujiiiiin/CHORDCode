#!/usr/bin/env python3
"""Assemble five approved asset pairs as CHORD meshes, prompts and CPU previews.

Run with the chord-mesh environment. No Gaussian fitting or motion training.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SPECS = {
    'cutting_board_with_croissant': dict(active=27, passive=10,
        prompt='The wooden cutting board tilts and presses down on the croissant, flattening the soft pastry and making its sides bulge, then lifts away while the croissant retains a shallow indentation.'),
    'baseball_with_pillow': dict(active=5, passive=22, select='throw_pillows_01_pillow01',
        prompt='The baseball moves downward into the soft pillow, creating a deep local dent, then rises away as the pillow slowly springs back to its original shape.'),
    'tyre_with_compost_bag': dict(active=19, passive=9, select='compost_bags_floor',
        prompt='The upright tire rolls forward over the filled compost bag, compressing the bag beneath it and pushing its contents outward, then rolls off as the bag settles into a flatter shape.'),
    'cutting_board_with_rubber_duck': dict(active=27, passive=20,
        prompt='The wooden cutting board lowers onto the rubber duck and gently squeezes its body, then lifts away as the duck gradually returns to its original shape.'),
    'broom_with_trashbag': dict(active=26, passive=23,
        prompt='The broom sweeps forward against the lower side of the filled trash bag, denting the bag at the contact point and pushing it a short distance across the floor, then draws back as the bag settles.'),
}


def build(name):
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector

    spec = SPECS[name]
    out = ROOT / 'data' / name
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 4

    def bounds(objects):
        points = []
        for obj in objects:
            vertices = np.empty(len(obj.data.vertices)*3, dtype=np.float64)
            obj.data.vertices.foreach_get('co', vertices)
            matrix = np.array(obj.matrix_world)
            points.append(vertices.reshape(-1,3) @ matrix[:3,:3].T + matrix[:3,3])
        points = np.concatenate(points)
        return points.min(0), points.max(0)

    def transform(objects, matrix):
        for obj in objects:
            obj.matrix_world = matrix @ obj.matrix_world
        bpy.context.view_layer.update()

    def load(asset_id, selection=None):
        path = next((ROOT / 'mydata/3Dassets/converted').glob(f'{asset_id:02d}_*')) / 'asset.glb'
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        new = set(bpy.data.objects) - before
        meshes = [o for o in new if o.type == 'MESH']
        if selection:
            meshes = [o for o in meshes if o.name == selection]
        if not meshes:
            raise ValueError(f'No selected meshes in {path}: {selection}')
        for obj in meshes:
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
        for obj in new - set(meshes):
            bpy.data.objects.remove(obj, do_unlink=True)
        return meshes, str(path.relative_to(ROOT))

    def place(objects, size, axis=0, rotation=None, x=0, y=0, bottom=0.002):
        if rotation:
            transform(objects, Matrix.Rotation(math.radians(rotation[1]), 4, rotation[0]))
        lo, hi = bounds(objects)
        scale = size / (hi[axis] - lo[axis])
        transform(objects, Matrix.Scale(float(scale), 4))
        lo, hi = bounds(objects)
        offset = (x - (lo[0] + hi[0])/2, y - (lo[1] + hi[1])/2, bottom - lo[2])
        transform(objects, Matrix.Translation(Vector(offset)))

    passive, passive_path = load(spec['passive'], spec.get('select'))
    active, active_path = load(spec['active'])
    if name == 'cutting_board_with_croissant':
        place(passive, .24)
        place(active, .34, rotation=('Y', 8), bottom=float(bounds(passive)[1][2]) + .025)
    elif name == 'baseball_with_pillow':
        # Source pillow is tilted; align its thin principal axis with vertical.
        points = np.array([o.matrix_world @ v.co for o in passive for v in o.data.vertices])
        values, vectors = np.linalg.eigh(np.cov(points.T))
        basis = vectors[:, [2, 1, 0]].T
        if np.linalg.det(basis) < 0:
            basis[0] *= -1
        matrix = np.eye(4)
        matrix[:3,:3] = basis
        transform(passive, Matrix(matrix.tolist()))
        place(passive, .50)
        place(active, .075, x=-.08, bottom=float(bounds(passive)[1][2]) + .055)
    elif name == 'tyre_with_compost_bag':
        place(passive, .50, x=.25)
        # The original tire already stands upright in the imported Z-up scene.
        place(active, .56, x=-.36)
    elif name == 'cutting_board_with_rubber_duck':
        place(passive, .24, axis=2)
        place(active, .32, bottom=float(bounds(passive)[1][2]) + .025)
    elif name == 'broom_with_trashbag':
        place(passive, .43, axis=2, x=.18)
        place(active, 1.00, axis=2, rotation=('Z', 90), x=-.25)

    groups = {'obj_0': active, 'obj_2': passive}
    lo, hi = bounds(active + passive)
    center = (lo + hi) / 2
    extent = float(max(hi - lo))
    # Neutral textured floor: texture-based CHORD loaders need a base-color map.
    bpy.ops.mesh.primitive_plane_add(size=extent*3.5, location=(center[0], center[1], 0))
    floor = bpy.context.object
    floor.name = 'static_floor'
    material = bpy.data.materials.new('neutral_floor')
    material.use_nodes = True
    image = bpy.data.images.new('neutral_floor_albedo', width=8, height=8)
    image.pixels[:] = [0.67, 0.65, 0.61, 1] * 64
    image.pack()
    texture = material.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image = image
    shader = material.node_tree.nodes.get('Principled BSDF')
    material.node_tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
    shader.inputs['Roughness'].default_value = .85
    floor.data.materials.append(material)
    groups['obj_1'] = [floor]

    def export(filename, objects):
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(out / filename), export_format='GLB',
            use_selection=True, export_apply=True, export_animations=False,
            export_cameras=False, export_lights=False, export_yup=True)

    export('scene.glb', active + passive)
    for key, objects in groups.items():
        export(key + '.glb', objects)
    (out / 'prompt.txt').write_text(spec['prompt'] + '\n')
    (out / 'negative_prompt.txt').write_text('.\n')
    metadata = dict(scene=name, assets=spec, active='obj_0', passive='obj_2', static='obj_1',
        source_paths={'obj_0': active_path, 'obj_2': passive_path},
        coordinate_system='GLB Y up (Blender Z up), metres; shared world transforms',
        objects={key: [dict(name=o.name, matrix_world_blender=[list(r) for r in o.matrix_world])
                      for o in objects] for key, objects in groups.items()},
        initial_bounds_blender={key: [v.tolist() for v in bounds(objects)]
                                for key, objects in groups.items()},
        scene_includes=['obj_0', 'obj_2'], preview_kind='static mesh render, not CHORD training output')
    (out / 'scene_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'README.md').write_text(f'''# {name}

Asset pair: {spec['active']:02d} + {spec['passive']:02d}.

- `obj_0.glb`: active object; `obj_2.glb`: passive object.
- `obj_1.glb`: static neutral floor.
- `scene.glb`: active + passive, without the floor (CHORD normalization reference).
- Shared world coordinates, GLB Y up, units in metres. No initial animation.
- `preview.png` / `preview_other.png`: static mesh views for composition review.
- `scene_metadata.json`: source assets, selected components and placement transforms.

## Motion prompt

{spec['prompt']}

Original CHORD prompts live in `example/*.sh` as `--prompt` arguments; see
`example/cat_with_cushion.sh`. These new text files are a convenience, not an
automatically discovered CHORD input. Pass their contents explicitly:

```bash
--mesh_source_path data/{name} --obj_num 3 \\
  --prompt "$(cat data/{name}/prompt.txt)" --n_prompt "."
```

The scene has not undergone Gaussian fitting, Wan supervision or MPM simulation.
Active/passive labels are metadata; CHORD does not infer physics roles from them.
Particle volume, constitutive properties and collision proxies remain separate
simulation setup steps. Broom bristles are thin alpha cards; rubber duck and
filled bags require appropriate volume/material approximations.
''')

    # Render actual exported composition with CPU Cycles, no GPU dependency.
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 720
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new('studio')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.8, .8, .8, 1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = .65
    bpy.ops.object.light_add(type='AREA', location=(center[0]-extent, center[1]-extent, extent*2.5))
    light = bpy.context.object
    light.data.energy = 150 * extent**2
    light.data.shape = 'DISK'
    light.data.size = extent*2
    light.rotation_euler = (Vector(center)-light.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = extent * 1.65
    for filename, azimuth, elevation in [('preview.png', -65, 22), ('preview_other.png', 35, 30)]:
        a,e=math.radians(azimuth),math.radians(elevation)
        camera.location = Vector(center) + Vector((math.cos(a)*math.cos(e), math.sin(a)*math.cos(e), math.sin(e))) * extent*3
        camera.rotation_euler = (Vector(center)-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath = str(out / filename)
        bpy.ops.render.render(write_still=True)
    print(f'COMPLETE {name}', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SPECS)
    args=parser.parse_args()
    if args.scene:
        build(args.scene)
        return
    for name in SPECS:
        out=ROOT/'data'/name
        out.mkdir(parents=True,exist_ok=True)
        with (out/'build.log').open('w') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--scene',name],stdout=log,stderr=subprocess.STDOUT)
        print(name,result.returncode,flush=True)
        if result.returncode:
            raise SystemExit(f'Failed: {out / "build.log"}')


if __name__ == '__main__':
    main()
