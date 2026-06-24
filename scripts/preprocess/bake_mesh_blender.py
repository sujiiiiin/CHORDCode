#!/usr/bin/env python3
"""Bake HDR environment lighting into mesh textures and export a baked GLB."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bpy
import numpy as np


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bake the appearance of a mesh under an HDR environment map into new "
            "textures so the exported mesh can be shown with raw texture rasterization."
        )
    )
    parser.add_argument(
        "--mesh_source_path",
        required=True,
        help="Directory containing the source mesh (defaults to scene.glb inside this directory).",
    )
    parser.add_argument(
        "--mesh_file",
        default=None,
        help="Optional explicit mesh file. If relative, it is resolved under --mesh_source_path.",
    )
    parser.add_argument(
        "--blender_env_exr",
        required=True,
        help="HDR environment map (.exr) used for the bake.",
    )
    parser.add_argument(
        "--output_mesh",
        default=None,
        help="Output baked GLB path. Defaults to <mesh_source_path>/baked/<stem>_baked.glb.",
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=0,
        help="Square bake resolution. If 0, infer from the source textures (fallback 2048).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Cycles samples for baking.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=16,
        help="Bake margin in pixels.",
    )
    parser.add_argument(
        "--world_strength",
        type=float,
        default=1.0,
        help="HDR world strength. Keep at 1.0 to match render_blender_glb_aligned.py.",
    )
    parser.add_argument(
        "--device",
        choices=("AUTO", "CUDA", "CPU"),
        default="AUTO",
        help="Bake device selection.",
    )
    parser.add_argument(
        "--alpha_threshold",
        type=float,
        default=0.999,
        help="Treat source alpha below this threshold as transparent and restore original base color there.",
    )
    return parser.parse_args(argv)


def ensure_object_mode() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def import_scene_generic(path: str) -> List[bpy.types.Object]:
    extension = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects.keys())
    if extension in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif extension == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    after = set(bpy.data.objects.keys())
    return [bpy.data.objects[name] for name in sorted(after - before)]


def resolve_mesh_path(args: argparse.Namespace) -> str:
    if args.mesh_file:
        mesh_path = args.mesh_file
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(args.mesh_source_path, mesh_path)
        if os.path.exists(mesh_path):
            return mesh_path
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    candidates = [
        "scene.glb",
        "scene.gltf",
        "scene.fbx",
    ]
    for candidate in candidates:
        mesh_path = os.path.join(args.mesh_source_path, candidate)
        if os.path.exists(mesh_path):
            return mesh_path
    raise FileNotFoundError(
        "Could not resolve a default mesh file under --mesh_source_path. "
        "Tried: scene.glb, scene.gltf, scene.fbx."
    )


def default_output_mesh(mesh_path: str, mesh_source_path: str) -> str:
    stem = os.path.splitext(os.path.basename(mesh_path))[0]
    out_dir = os.path.join(mesh_source_path, "baked")
    return os.path.join(out_dir, f"{stem}_baked.glb")


def setup_hdr_lighting(env_exr: str, strength: float) -> None:
    if not os.path.exists(env_exr):
        raise FileNotFoundError(f"Environment EXR not found: {env_exr}")

    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(env_exr)
    try:
        env_tex.image.colorspace_settings.name = "Raw"
    except TypeError:
        env_tex.image.colorspace_settings.name = "Non-Color"
    env_tex.location = (-600, 0)

    background_env = nodes.new(type="ShaderNodeBackground")
    background_env.inputs["Strength"].default_value = strength
    background_env.location = (-350, 120)

    # Keep HDRI directional lighting, but avoid mirroring HDR details in glossy reflections.
    background_reflection = nodes.new(type="ShaderNodeBackground")
    background_reflection.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background_reflection.inputs["Strength"].default_value = strength
    background_reflection.location = (-350, -120)

    light_path = nodes.new(type="ShaderNodeLightPath")
    light_path.location = (-600, -220)

    reflection_mix = nodes.new(type="ShaderNodeMath")
    reflection_mix.operation = "MAXIMUM"
    reflection_mix.location = (-380, -180)

    mix_shader = nodes.new(type="ShaderNodeMixShader")
    mix_shader.location = (-160, 0)

    output = nodes.new(type="ShaderNodeOutputWorld")
    output.location = (60, 0)

    links.new(env_tex.outputs["Color"], background_env.inputs["Color"])
    links.new(light_path.outputs["Is Glossy Ray"], reflection_mix.inputs[0])
    links.new(light_path.outputs["Is Transmission Ray"], reflection_mix.inputs[1])
    links.new(reflection_mix.outputs["Value"], mix_shader.inputs["Fac"])
    links.new(background_env.outputs["Background"], mix_shader.inputs[1])
    links.new(background_reflection.outputs["Background"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])


def configure_cycles(device: str, samples: int, margin: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = max(int(samples), 1)
    scene.cycles.preview_samples = min(scene.cycles.samples, 64)
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.bake.target = "IMAGE_TEXTURES"
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.use_clear = True
    scene.render.bake.margin = max(int(margin), 0)
    scene.render.bake.use_pass_direct = True
    scene.render.bake.use_pass_indirect = True
    scene.render.bake.use_pass_color = True

    if device == "CPU":
        scene.cycles.device = "CPU"
        return

    try:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons["cycles"].preferences
        cycles_prefs.compute_device_type = "CUDA"
        cycles_prefs.get_devices()
        for dev in cycles_prefs.devices:
            dev.use = device != "AUTO" or dev.type != "CPU"
        scene.cycles.device = "GPU"
    except Exception:
        scene.cycles.device = "CPU"


def infer_material_size(material: Optional[bpy.types.Material], fallback: int) -> Tuple[int, int]:
    if material is None or not material.use_nodes or material.node_tree is None:
        return fallback, fallback

    max_w = 0
    max_h = 0
    for node in material.node_tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        size = tuple(node.image.size)
        if len(size) >= 2 and size[0] > 0 and size[1] > 0:
            max_w = max(max_w, int(size[0]))
            max_h = max(max_h, int(size[1]))
    if max_w > 0 and max_h > 0:
        return max_w, max_h
    return fallback, fallback


def find_principled(material: Optional[bpy.types.Material]) -> Optional[bpy.types.ShaderNodeBsdfPrincipled]:
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    return None


def find_upstream_image(socket: bpy.types.NodeSocket, depth: int = 8) -> Optional[bpy.types.Image]:
    if depth <= 0 or socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        node = link.from_node
        if node is None:
            continue
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node.image
        for inp in node.inputs:
            image = find_upstream_image(inp, depth - 1)
            if image is not None:
                return image
    return None


def image_to_numpy_rgba(image: Optional[bpy.types.Image]) -> Optional[np.ndarray]:
    if image is None or image.size[0] <= 0 or image.size[1] <= 0:
        return None
    try:
        image.update()
    except RuntimeError:
        return None
    flat = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(flat)
    return flat.reshape((image.size[1], image.size[0], 4))


def resize_rgba_nearest(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = rgba.shape[:2]
    if src_w == width and src_h == height:
        return rgba
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(np.int64), 0, src_w - 1)
    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(np.int64), 0, src_h - 1)
    return rgba[y_idx[:, None], x_idx[None, :], :]


def resolve_source_color_and_alpha(material: bpy.types.Material, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    principled = find_principled(material)
    base_rgb = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    base_alpha = np.float32(1.0)
    base_image = None
    alpha_image = None

    if principled is not None:
        if "Base Color" in principled.inputs:
            base_rgb = np.array(principled.inputs["Base Color"].default_value[:3], dtype=np.float32)
            base_image = find_upstream_image(principled.inputs["Base Color"])
        if "Alpha" in principled.inputs:
            base_alpha = np.float32(principled.inputs["Alpha"].default_value)
            alpha_image = find_upstream_image(principled.inputs["Alpha"])

    base_rgba = image_to_numpy_rgba(base_image)
    if base_rgba is not None:
        base_rgba = resize_rgba_nearest(base_rgba, width, height)
        source_rgb = np.asarray(base_rgba[..., :3], dtype=np.float32)
        source_alpha = np.asarray(base_rgba[..., 3], dtype=np.float32)
    else:
        source_rgb = np.ones((height, width, 3), dtype=np.float32) * base_rgb.reshape(1, 1, 3)
        source_alpha = np.ones((height, width), dtype=np.float32) * base_alpha

    alpha_rgba = image_to_numpy_rgba(alpha_image)
    if alpha_rgba is not None:
        alpha_rgba = resize_rgba_nearest(alpha_rgba, width, height)
        source_alpha = np.asarray(alpha_rgba[..., 0], dtype=np.float32)

    return source_rgb, source_alpha


def material_has_transparency(material: bpy.types.Material, source_alpha: np.ndarray, alpha_threshold: float) -> bool:
    if material.blend_method in {"BLEND", "HASHED", "CLIP"}:
        return True

    principled = find_principled(material)
    if principled is None:
        return bool(np.any(source_alpha < np.float32(alpha_threshold)))

    if "Alpha" in principled.inputs and float(principled.inputs["Alpha"].default_value) < 0.999:
        return True

    for transmission_name in ("Transmission Weight", "Transmission"):
        if transmission_name in principled.inputs and float(principled.inputs[transmission_name].default_value) > 1e-4:
            return True

    return bool(np.any(source_alpha < np.float32(alpha_threshold)))


def apply_transparent_basecolor_fallback(
    bake_image: bpy.types.Image,
    source_rgb: np.ndarray,
    source_alpha: np.ndarray,
    transparent_material: bool,
    alpha_threshold: float,
) -> None:
    baked_rgba = image_to_numpy_rgba(bake_image)
    if baked_rgba is None:
        return

    baked_rgb = np.asarray(baked_rgba[..., :3], dtype=np.float32)
    if transparent_material:
        alpha_mask = source_alpha < np.float32(alpha_threshold)
        if np.any(alpha_mask):
            baked_rgb[alpha_mask] = source_rgb[alpha_mask]
        elif source_alpha.shape == baked_rgb.shape[:2]:
            # Pure transmission/glass materials often have alpha=1 everywhere; treat entire slot as transparent.
            baked_rgb[:, :] = source_rgb

    out_rgba = np.ones_like(baked_rgba, dtype=np.float32)
    out_rgba[..., :3] = np.clip(baked_rgb, 0.0, 1.0)
    out_rgba[..., 3] = 1.0
    bake_image.pixels.foreach_set(out_rgba.reshape(-1))
    bake_image.update()


def ensure_uvs(mesh_obj: bpy.types.Object) -> None:
    if mesh_obj.type != "MESH" or len(mesh_obj.data.uv_layers) > 0:
        return
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def make_material_slot_unique(mesh_obj: bpy.types.Object, slot_idx: int) -> bpy.types.Material:
    slot = mesh_obj.material_slots[slot_idx]
    base = slot.material
    if base is None:
        base = bpy.data.materials.new(f"{mesh_obj.name}_Mat_{slot_idx}")
        base.use_nodes = True
    material = base.copy()
    material.name = f"{mesh_obj.name}_Baked_{slot_idx}"
    slot.material = material
    return material


def prepare_bake_target(
    material: bpy.types.Material,
    image_name: str,
    width: int,
    height: int,
) -> Tuple[bpy.types.Image, bpy.types.ShaderNodeTexImage]:
    if not material.use_nodes or material.node_tree is None:
        material.use_nodes = True

    nodes = material.node_tree.nodes
    bake_image = bpy.data.images.new(image_name, width=width, height=height, alpha=False)
    bake_image.generated_color = (1.0, 1.0, 1.0, 1.0)
    bake_image.file_format = "PNG"

    image_node = nodes.new(type="ShaderNodeTexImage")
    image_node.name = "__BakeTarget"
    image_node.label = "__BakeTarget"
    image_node.image = bake_image
    image_node.interpolation = "Linear"
    image_node.select = True
    nodes.active = image_node
    return bake_image, image_node


def finalize_baked_material(material: bpy.types.Material, bake_image: bpy.types.Image) -> None:
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    tex = nodes.new(type="ShaderNodeTexImage")
    tex.location = (-300, 0)
    tex.image = bake_image
    tex.interpolation = "Linear"

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.location = (-50, 0)
    if "Roughness" in principled.inputs:
        principled.inputs["Roughness"].default_value = 1.0
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (200, 0)

    links.new(tex.outputs["Color"], principled.inputs["Base Color"])
    material.blend_method = "OPAQUE"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "OPAQUE"
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])


def bake_object(
    mesh_obj: bpy.types.Object,
    fallback_size: int,
    alpha_threshold: float,
) -> List[bpy.types.Image]:
    ensure_uvs(mesh_obj)

    slot_infos: List[Tuple[bpy.types.Material, bpy.types.Image, np.ndarray, np.ndarray, bool]] = []
    baked_images: List[bpy.types.Image] = []
    for slot_idx in range(len(mesh_obj.material_slots)):
        material = make_material_slot_unique(mesh_obj, slot_idx)
        width, height = infer_material_size(material, fallback_size)
        source_rgb, source_alpha = resolve_source_color_and_alpha(material, width, height)
        transparent_material = material_has_transparency(material, source_alpha, alpha_threshold)
        bake_image, _ = prepare_bake_target(
            material,
            image_name=f"{mesh_obj.name}_slot{slot_idx}_baked",
            width=width,
            height=height,
        )
        slot_infos.append((material, bake_image, source_rgb, source_alpha, transparent_material))
        baked_images.append(bake_image)

    if not baked_images:
        return baked_images

    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.bake(type="COMBINED")

    for material, image, source_rgb, source_alpha, transparent_material in slot_infos:
        apply_transparent_basecolor_fallback(
            image,
            source_rgb,
            source_alpha,
            transparent_material=transparent_material,
            alpha_threshold=alpha_threshold,
        )
        try:
            image.pack()
        except RuntimeError:
            pass
        finalize_baked_material(material, image)

    return baked_images


def export_baked_mesh(output_mesh: str, exported_objects: Iterable[bpy.types.Object]) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in exported_objects:
        if obj.name in bpy.data.objects:
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=output_mesh,
        export_format="GLB",
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    mesh_path = resolve_mesh_path(args)
    output_mesh = args.output_mesh or default_output_mesh(mesh_path, args.mesh_source_path)
    output_mesh = os.path.abspath(output_mesh)

    fallback_size = args.texture_size if args.texture_size > 0 else 2048

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ensure_object_mode()
    setup_hdr_lighting(args.blender_env_exr, args.world_strength)
    configure_cycles(args.device, args.samples, args.margin)

    imported_objects = import_scene_generic(mesh_path)
    mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")

    total_images = 0
    for mesh_obj in mesh_objects:
        total_images += len(bake_object(mesh_obj, fallback_size, args.alpha_threshold))

    if total_images == 0:
        raise RuntimeError("Nothing was baked: imported mesh had no material slots.")

    os.makedirs(os.path.dirname(output_mesh), exist_ok=True)
    export_baked_mesh(output_mesh, imported_objects)

    print(f"Imported mesh: {mesh_path}")
    print(f"Baked texture count: {total_images}")
    print(f"Saved baked mesh: {output_mesh}")


if __name__ == "__main__":
    main()
