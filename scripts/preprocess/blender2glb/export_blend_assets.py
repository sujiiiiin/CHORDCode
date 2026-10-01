#!/usr/bin/env python3
"""Extract zipped Blender assets and export textured GLB and OBJ in isolation.

Run with the chord-mesh environment. Each asset gets its own subprocess, log,
and JSON report. Originals are preserved; no CHORD scene is assembled here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import zipfile


def validate(path):
    import numpy as np
    import trimesh

    scene = trimesh.load_scene(path, process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    if not meshes or any(not len(m.faces) for m in meshes):
        raise ValueError(f"Empty mesh: {path}")
    if any(not np.isfinite(m.vertices).all() for m in meshes):
        raise ValueError(f"Nonfinite vertices: {path}")
    joined = trimesh.util.concatenate(meshes)
    # Exported vertices are split at UV seams; weld a copy for topology checks.
    topology = trimesh.Trimesh(vertices=joined.vertices, faces=joined.faces, process=True)
    return dict(vertices=len(joined.vertices), triangles=len(joined.faces),
                bounds=joined.bounds.tolist(), geometry_parts=len(meshes),
                watertight_after_welding=bool(topology.is_watertight),
                winding_consistent=bool(topology.is_winding_consistent),
                uv_parts=sum(getattr(m.visual, 'uv', None) is not None for m in meshes))


def validate_textures(glb, obj):
    with glb.open('rb') as f:
        magic, version, total = struct.unpack('<4sII', f.read(12))
        size, kind = struct.unpack('<II', f.read(8))
        if magic != b'glTF' or version != 2 or kind != 0x4E4F534A:
            raise ValueError('Invalid GLB header')
        document = json.loads(f.read(size))
    images = document.get('images', [])
    if any('bufferView' not in image for image in images):
        raise ValueError('GLB has nonembedded images')
    materials = document.get('materials', [])
    textured = sum('baseColorTexture' in m.get('pbrMetallicRoughness', {}) for m in materials)
    basecolor_images = []
    for material in materials:
        texture = material.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if texture:
            source = document['textures'][texture['index']]['source']
            name = images[source].get('name', '')
            if '_diff' not in name.lower():
                raise ValueError(f'Unexpected base color image: {name}')
            basecolor_images.append(name)
    mtl = obj.with_suffix('.mtl')
    maps = []
    diffuse_maps = 0
    for line in mtl.read_text().splitlines():
        if line.startswith(('map_', 'bump ', 'disp ')):
            value = line.split(maxsplit=1)[1]
            if value.startswith('-bm '):
                value = value.split(maxsplit=2)[2]
            target = obj.parent / value
            if not target.is_file():
                raise FileNotFoundError(f'Missing OBJ texture: {target}')
            maps.append(target.name)
            diffuse_maps += line.startswith('map_Kd ')
    if textured != len(materials) or diffuse_maps != len(materials):
        raise ValueError('Some exported materials lack diffuse textures')
    return dict(embedded_glb_images=len(images), glb_materials=len(materials),
                glb_basecolor_textured_materials=textured,
                glb_basecolor_images=basecolor_images,
                obj_diffuse_maps=diffuse_maps, obj_texture_files=maps)


def convert(blend, output):
    import bpy

    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 4
    # Excluded collections contain alternate LOD meshes, not additional parts.
    objects = [o for o in bpy.context.view_layer.objects
               if o.type == 'MESH' and o.visible_get() and not o.hide_render]
    excluded = [o.name for o in bpy.context.scene.objects
                if o.type == 'MESH' and o not in objects]
    if not objects:
        raise ValueError('No mesh objects in active scene')
    def material_images(tree, seen=None):
        seen = set() if seen is None else seen
        if tree is None or tree in seen:
            return set()
        seen.add(tree)
        found = set()
        for node in tree.nodes:
            if getattr(node, 'image', None):
                found.add(node.image)
            if node.type == 'GROUP':
                found.update(material_images(node.node_tree, seen))
        return found
    used_images = set()
    for obj in objects:
        for material in obj.data.materials:
            if material:
                used_images.update(material_images(material.node_tree))
    images = []
    for img in sorted(used_images, key=lambda i: i.name):
        if img.source != 'FILE':
            continue
        source = Path(bpy.path.abspath(img.filepath))
        if not source.is_file() and not img.packed_file:
            matches = list(blend.parent.rglob(source.name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Missing/ambiguous image: {img.filepath}")
            source = matches[0]
        if not img.packed_file:
            img.filepath = str(source)
            img.reload()
        images.append(dict(name=img.name, size=list(img.size), source=str(source)))
        if not all(img.size):
            raise ValueError(f"Unreadable image: {source}")
    # glTF/OBJ cannot represent Blender color-mix graphs reliably. Use the
    # supplied scan albedo explicitly, rather than accidentally exporting a mask.
    albedo_rewired = []
    materials = {m for o in objects for m in o.data.materials if m}
    for material in materials:
        candidates = [i for i in material_images(material.node_tree)
                      if '_diff' in Path(i.filepath).name.lower()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one albedo texture for {material.name}')
        shaders = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
        if len(shaders) != 1:
            raise ValueError(f'Expected one Principled shader for {material.name}')
        socket = shaders[0].inputs['Base Color']
        if (len(socket.links) == 1 and socket.links[0].from_node.type == 'TEX_IMAGE'
                and socket.links[0].from_node.image == candidates[0]):
            continue
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.image = candidates[0]
        material.node_tree.links.new(node.outputs['Color'], socket)
        albedo_rewired.append(material.name)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    object_info = [dict(name=o.name, vertices=len(o.data.vertices),
                        polygons=len(o.data.polygons),
                        materials=[m.name if m else None for m in o.data.materials],
                        modifiers=[m.type for m in o.modifiers]) for o in objects]
    for obj in objects:
        obj.modifiers.new(name='CHORD export triangulation', type='TRIANGULATE')
    glb = output / 'asset.glb'
    obj = output / 'asset.obj'
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB',
        use_selection=True, export_apply=True, export_yup=True,
        export_animations=False, export_cameras=False, export_lights=False)
    # Both formats use Y up, -Z forward. OBJ has no standard embedded textures.
    bpy.ops.wm.obj_export(filepath=str(obj), export_selected_objects=True,
        apply_modifiers=True, export_triangulated_mesh=True,
        forward_axis='NEGATIVE_Z', up_axis='Y', path_mode='COPY',
        export_materials=True)
    # Release Blender's decoded 4K textures before independently loading exports.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    glb_info = validate(glb)
    obj_info = validate(obj)
    textures = validate_textures(glb, obj)
    import numpy as np
    if not np.allclose(glb_info['bounds'], obj_info['bounds'], atol=1e-5, rtol=1e-5):
        raise ValueError('GLB/OBJ bounds mismatch')
    if glb_info['triangles'] != obj_info['triangles']:
        raise ValueError('GLB/OBJ triangle count mismatch')
    report = dict(blender=bpy.app.version_string, source=str(blend),
                  axis='Y up, -Z forward; original scale and object transforms',
                  objects=object_info, excluded_objects=excluded,
                  albedo_rewired_to_source_diffuse=albedo_rewired,
                  images=images, textures=textures,
                  glb=glb_info, obj=obj_info)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('mydata/3Dassets'))
    parser.add_argument('--only', help='Archive name prefix for a pilot conversion')
    parser.add_argument('--resume', action='store_true', help='Reuse completed reports')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        convert(args.worker.resolve(), args.output.resolve())
        return
    root = args.source.resolve()
    rows = []
    for index, archive in enumerate(sorted(root.glob('*.blend.zip')), 1):
        name = archive.name.removesuffix('_4k.blend.zip')
        if args.only and not archive.name.startswith(args.only):
            continue
        extracted = root / 'extracted' / name
        output = root / 'converted' / f'{index:02d}_{name}'
        extracted.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        if args.resume and (output / 'report.json').exists():
            rows.append(dict(id=index, name=name, archive=archive.name,
                output=str(output.relative_to(root)), returncode=0,
                report=json.loads((output / 'report.json').read_text())))
            continue
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (extracted / member.filename).resolve()
                if not target.is_relative_to(extracted):
                    raise ValueError(f'Unsafe archive path: {member.filename}')
            z.extractall(extracted)
        blends = list(extracted.glob('*.blend'))
        if len(blends) != 1:
            raise ValueError(f'Expected one blend in {archive}')
        print(f'[{index:02d}] {name}', flush=True)
        with (output / 'conversion.log').open('w') as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--worker', str(blends[0]), '--output', str(output)],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
        row = dict(id=index, name=name, archive=archive.name,
                   output=str(output.relative_to(root)), returncode=result.returncode)
        if result.returncode == 0:
            row['report'] = json.loads((output / 'report.json').read_text())
        else:
            print(f'FAILED ({result.returncode}): see {output / "conversion.log"}', flush=True)
        rows.append(row)
        (root / ('manifest_pilot.json' if args.only else 'manifest.json')).write_text(
            json.dumps(rows, indent=2) + '\n')
    (root / ('manifest_pilot.json' if args.only else 'manifest.json')).write_text(
        json.dumps(rows, indent=2) + '\n')
    if not rows or any(r['returncode'] for r in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
