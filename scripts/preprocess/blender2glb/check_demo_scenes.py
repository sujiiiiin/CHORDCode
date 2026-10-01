#!/usr/bin/env python3
"""Check CHORD scene splits, shared coordinates, textures and initial separation."""
import json
from pathlib import Path
import struct

import numpy as np
import trimesh

from build_demo_scenes import ROOT, SPECS


def read_mesh(path):
    scene = trimesh.load_scene(path, process=False, skip_materials=True)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes)
    assert len(mesh.faces) and np.isfinite(mesh.vertices).all(), path
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        length, kind = struct.unpack('<II', stream.read(8))
        document = json.loads(stream.read(length))
    assert magic == b'glTF' and version == 2 and size == path.stat().st_size
    assert all('bufferView' in image for image in document['images'])
    assert all('baseColorTexture' in m.get('pbrMetallicRoughness', {})
               for m in document['materials']), path
    return mesh


def main():
    for name, spec in SPECS.items():
        root = ROOT / 'data' / name
        meshes = {key: read_mesh(root / (key + '.glb'))
                  for key in ['scene', 'obj_0', 'obj_1', 'obj_2']}
        active, passive, floor = (meshes[k] for k in ['obj_0', 'obj_2', 'obj_1'])
        combined = trimesh.util.concatenate([active, passive])
        assert len(combined.faces) == len(meshes['scene'].faces), name
        # Shared world coordinates, independent of exporter vertex order/splitting.
        a = np.unique(np.round(combined.vertices, 5), axis=0)
        b = np.unique(np.round(meshes['scene'].vertices, 5), axis=0)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5), name
        assert np.allclose(floor.bounds[:, 1], 0, atol=1e-6), name
        assert combined.bounds[0, 1] >= -1e-6, name
        assert abs(passive.bounds[0, 1] - .002) < 1e-5, (name, 'passive not resting at ground clearance')
        axis = 0 if name in ['tyre_with_compost_bag', 'broom_with_trashbag'] else 1
        gap = (passive.bounds[0, 0] - active.bounds[1, 0] if axis == 0
               else active.bounds[0, 1] - passive.bounds[1, 1])
        assert gap > 0, (name, 'initial bounding boxes overlap', gap)
        assert (root / 'prompt.txt').read_text().strip() == spec['prompt']
        assert (root / 'negative_prompt.txt').read_text().strip() == '.'
        assert (root / 'preview.png').is_file() and (root / 'preview_other.png').is_file()
        result = dict(passed=True, shared_world_coordinates=True,
            scene_is_active_plus_passive=True, textures_embedded=True,
            no_initial_active_passive_overlap=True, separation_axis='XYZ'[axis],
            initial_separation_metres=float(gap),
            bounds_glb={k: m.bounds.tolist() for k, m in meshes.items()},
            triangles={k: len(m.faces) for k, m in meshes.items()})
        (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name, f'PASS; initial gap {gap:.4f} m', flush=True)


if __name__ == '__main__':
    main()
