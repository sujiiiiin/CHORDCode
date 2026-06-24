#!/usr/bin/env python3
"""Render an animated glTF/FBX asset under HDR lighting with camera placement derived from a reference scene mesh."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROGRESS_FD = os.dup(1)


@contextlib.contextmanager
def suppress_native_output():
    """Silence Blender/native-library writes that bypass Python stdout objects."""
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


with suppress_native_output():
    import bpy
    import mathutils


def print_progress_line(text: str, final: bool = False) -> None:
    suffix = "\n" if final else "\r"
    os.write(PROGRESS_FD, (text + suffix).encode("utf-8", errors="replace"))


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render animated glTF/FBX with HDR lighting")
    parser.add_argument("--mesh_file", required=True, help="Path to animated mesh (.glb/.gltf/.fbx)")
    parser.add_argument("--scene_mesh", required=True, help="Reference scene glTF/GLB for centre/scale")
    parser.add_argument("--env_exr", required=True, help="HDR environment map (.exr) used for lighting (white background)")
    parser.add_argument("--output", required=True, help="Output directory for renders")
    parser.add_argument("--save-video", action="store_true", help="Render to a single video file instead of image sequence")
    parser.add_argument("--save-frames", action="store_true", help="Save image frames in addition to video")
    parser.add_argument("--test", action="store_true", help="Render only the first frame for testing camera placement")
    parser.add_argument("--camera-name", default=None, help="Name of camera to use/create (optional)")
    parser.add_argument("--resolution-x", type=int, default=1920, help="Render width in pixels")
    parser.add_argument("--resolution-y", type=int, default=1080, help="Render height in pixels")
    parser.add_argument("--camera-fov", type=float, default=49.1, help="Camera field of view in degrees")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--elev", type=float, default=15.0, help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=45.0, help="Camera azimuth in degrees")
    parser.add_argument("--cam-radius", type=float, default=2.0, help="Camera radius in normalised units")
    parser.add_argument("--render-frame", type=int, default=None, help="Render only the specified frame index")
    parser.add_argument(
        "--render_no_texture",
        action="store_true",
        help="Override imported materials with a flat grey surface",
    )
    parser.add_argument(
        "--add_white_floor",
        action="store_true",
        help="Add a large white floor plane beneath the objects to catch shadows",
    )
    parser.add_argument(
        "--external_ground",
        default=None,
        help="Path to an external ground mesh (.glb/.gltf/.fbx) placed beneath the scene",
    )
    parser.add_argument(
        "--only_shadow",
        action="store_true",
        help="When used with --add_white_floor or --external_ground, make the ground act as a shadow catcher",
    )
    parser.add_argument(
        "--transparent-bg",
        action="store_true",
        help="Render with transparent background (environment map still contributes lighting/reflections)",
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ensure_object_mode() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def import_scene_generic(path: str) -> List[bpy.types.Object]:
    extension = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects.keys())
    with suppress_native_output():
        if extension in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=path)
        elif extension == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            raise ValueError(f"Unsupported mesh format: {path}")
    after = set(bpy.data.objects.keys())
    new_names = sorted(after - before)
    new_objects = [bpy.data.objects[name] for name in new_names]
    for obj in new_objects:
        if obj and hasattr(obj, "use_evaluation_mode"):
            obj.use_evaluation_mode = "EVAL"
    return new_objects


def cleanup_objects(objects: Iterable[bpy.types.Object]) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        if obj and obj.name in bpy.data.objects:
            obj.select_set(True)
    if any(obj.select_get() for obj in objects if obj):
        bpy.ops.object.delete(use_global=False)

    for datablock in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures):
        for block in list(datablock):
            if block.users == 0:
                datablock.remove(block, do_unlink=True)


def compute_bounds(objects: Iterable[bpy.types.Object]) -> Tuple[mathutils.Vector, float, float, mathutils.Vector, mathutils.Vector]:
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh geometry found to compute bounds.")

    vmin = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    vmax = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in mesh_objects:
        matrix = obj.matrix_world
        for corner in obj.bound_box:
            world_corner = matrix @ mathutils.Vector(corner)
            vmin.x = min(vmin.x, world_corner.x)
            vmin.y = min(vmin.y, world_corner.y)
            vmin.z = min(vmin.z, world_corner.z)
            vmax.x = max(vmax.x, world_corner.x)
            vmax.y = max(vmax.y, world_corner.y)
            vmax.z = max(vmax.z, world_corner.z)

    extent = vmax - vmin
    max_extent = max(extent.x, extent.y, extent.z, 1e-9)
    center = (vmin + vmax) * 0.5
    scale = 1.2 / max_extent
    return center, scale, vmin.z, vmin.copy(), vmax.copy()


def compute_reference_center_scale(scene_mesh_path: str) -> Tuple[mathutils.Vector, float, float]:
    imported = import_scene_generic(scene_mesh_path)
    try:
        center, scale, min_z, _, _ = compute_bounds(imported)
    finally:
        cleanup_objects(imported)
    return center, scale, min_z


def apply_untextured_material(objects: Iterable[bpy.types.Object]) -> None:
    material_name = "__RenderGreyMaterial"
    grey = bpy.data.materials.get(material_name)
    if grey is None:
        grey = bpy.data.materials.new(material_name)
        grey.use_nodes = True
        nodes = grey.node_tree.nodes
        links = grey.node_tree.links
        nodes.clear()
        principled = nodes.new(type="ShaderNodeBsdfPrincipled")
        principled.location = (-200, 0)
        principled.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1.0)
        principled.inputs["Roughness"].default_value = 0.4
        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (0, 0)
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        grey.blend_method = "OPAQUE"

    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        while mesh.materials:
            mesh.materials.pop(index=len(mesh.materials) - 1)
        mesh.materials.append(grey)


def find_or_create_camera(args: argparse.Namespace) -> bpy.types.Object:
    scene = bpy.context.scene
    cam = None
    if args.camera_name:
        cam = bpy.data.objects.get(args.camera_name)
        if cam and cam.type != "CAMERA":
            cam = None

    if cam is None:
        cam = scene.camera

    if cam is None or cam.type != "CAMERA":
        camera_data = bpy.data.cameras.new(args.camera_name or "RenderCamera")
        cam = bpy.data.objects.new(camera_data.name, camera_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
    else:
        scene.camera = cam

    cam.data.clip_end = max(cam.data.clip_end, 1000.0)
    return cam


def position_camera(cam: bpy.types.Object, center: mathutils.Vector, scale: float, args: argparse.Namespace) -> None:
    center_vec = center.copy()
    radius_world = args.cam_radius / scale
    elev = math.radians(args.elev)
    azim = math.radians(args.azim)

    x = center_vec.x + radius_world * math.cos(elev) * math.cos(azim)
    y = center_vec.y + radius_world * math.cos(elev) * math.sin(azim)
    z = center_vec.z + radius_world * math.sin(elev)

    cam.location = (x, y, z)
    direction = center_vec - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_end = max(cam.data.clip_end, radius_world * 10.0)


def configure_camera_fov(cam: bpy.types.Object, args: argparse.Namespace) -> None:
    camera_data = cam.data
    if camera_data.type != "PERSP":
        camera_data.type = "PERSP"
    camera_data.angle = math.radians(args.camera_fov)


def configure_cycles_cuda() -> None:
    try:
        with suppress_native_output():
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons["cycles"].preferences
            cycles_prefs.compute_device_type = "CUDA"
            cycles_prefs.get_devices()
            for device in cycles_prefs.devices:
                device.use = device.type != "CPU"
            bpy.context.scene.cycles.device = "GPU"
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"Warning: unable to configure CUDA rendering automatically: {exc}")


def save_debug_blend(reference_path: str) -> str:
    directory, filename = os.path.split(os.path.abspath(reference_path))
    stem, _ = os.path.splitext(filename)
    debug_path = os.path.join(directory, f"{stem}_camera.blend")
    try:
        with suppress_native_output():
            bpy.ops.wm.save_as_mainfile(filepath=debug_path, copy=True)
    except RuntimeError as exc:
        print(f"Warning: failed to save debug blend '{debug_path}': {exc}")
        return ""
    return debug_path


def _find_last_animated_frame(actions: Iterable[bpy.types.Action], start: int, end: int, epsilon: float = 1e-6) -> int:
    last_change = start
    for action in actions:
        for fcurve in action.fcurves:
            points = list(fcurve.keyframe_points)
            if not points and fcurve.sampled_points:
                points = list(fcurve.sampled_points)
            if not points:
                continue
            points.sort(key=lambda pt: float(pt.co[0]))
            prev_point = points[0]
            prev_value = float(prev_point.co[1])
            for point in points[1:]:
                frame = float(point.co[0])
                value = float(point.co[1])
                value_changed = abs(value - prev_value) > epsilon
                if not value_changed:
                    handle_variation = False
                    if hasattr(prev_point, "handle_right"):
                        handle_variation = handle_variation or abs(prev_point.handle_right.y - prev_value) > epsilon
                    if hasattr(point, "handle_left"):
                        handle_variation = handle_variation or abs(point.handle_left.y - value) > epsilon
                    if handle_variation:
                        value_changed = True
                if value_changed:
                    last_change = max(last_change, int(math.floor(frame)))
                prev_point = point
                prev_value = value
    return last_change


def determine_frame_range() -> Tuple[int, int]:
    scene = bpy.context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)

    actions = list(bpy.data.actions)
    if actions:
        action_start = min(int(action.frame_range[0]) for action in actions)
        action_end = max(int(action.frame_range[1]) for action in actions)
        start = min(start, action_start)
        end = max(end, action_end)
        trimmed_end = _find_last_animated_frame(actions, start, end)
        if trimmed_end < end:
            end = max(trimmed_end, start)

    if start >= end:
        end = start
    return start, end


def configure_render_resolution(scene: bpy.types.Scene, args: argparse.Namespace) -> None:
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0


def configure_render_frames(args: argparse.Namespace, frame_range: Tuple[int, int], frames_dir: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    configure_render_resolution(scene, args)
    scene.render.fps = args.fps
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = bool(getattr(args, "only_shadow", False) or getattr(args, "transparent_bg", False))
    scene.cycles.film_transparent = scene.render.film_transparent

    scene.frame_start, scene.frame_end = frame_range
    if args.test:
        scene.frame_end = scene.frame_start
    scene.frame_current = scene.frame_start
    scene.frame_set(scene.frame_start)

    os.makedirs(frames_dir, exist_ok=True)
    scene.render.filepath = os.path.join(frames_dir, "frame_")


def configure_render_video(args: argparse.Namespace, frame_range: Tuple[int, int], video_path: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    configure_render_resolution(scene, args)
    scene.render.fps = args.fps
    scene.render.use_file_extension = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.film_transparent = bool(getattr(args, "only_shadow", False) or getattr(args, "transparent_bg", False))
    scene.cycles.film_transparent = scene.render.film_transparent

    scene.frame_start, scene.frame_end = frame_range
    if args.test:
        scene.frame_end = scene.frame_start
    scene.frame_current = scene.frame_start
    scene.frame_set(scene.frame_start)

    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scene.render.ffmpeg.gopsize = scene.render.fps * 2
    scene.render.ffmpeg.audio_codec = 'NONE'

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    scene.render.filepath = video_path


def render_quiet(target_desc: str) -> None:
    scene = bpy.context.scene
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    total = max(end - start + 1, 1)
    rendered_count = 0
    last_progress_text = None

    def _format_progress(count: int, frame: int) -> str:
        return (
            f"Rendering frames: {count}/{total} "
            f"(frame {frame}, output: {target_desc})"
        )

    def _print_progress(frame: int, count: int, final: bool = False) -> None:
        nonlocal last_progress_text
        count = min(max(count, 0), total)
        text = _format_progress(count, frame)
        if text == last_progress_text and not final:
            return
        if text == last_progress_text and final:
            print_progress_line("", final=True)
            return
        print_progress_line(text, final=final)
        last_progress_text = text

    def _on_frame_change(render_scene):
        frame = int(render_scene.frame_current)
        frame_offset = min(max(frame - start, 0), total - 1)
        _print_progress(frame, max(rendered_count, frame_offset))

    def _on_frame_written(render_scene):
        nonlocal rendered_count
        rendered_count = min(rendered_count + 1, total)
        frame = int(render_scene.frame_current)
        _print_progress(frame, rendered_count)

    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    bpy.app.handlers.render_write.append(_on_frame_written)
    try:
        _print_progress(start, 0)
        with suppress_native_output():
            bpy.ops.render.render(animation=True)
    finally:
        if _on_frame_change in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(_on_frame_change)
        if _on_frame_written in bpy.app.handlers.render_write:
            bpy.app.handlers.render_write.remove(_on_frame_written)

    _print_progress(end, total, final=True)
    print(f"Render completed: {target_desc}")


def setup_hdr_lighting(env_exr: str) -> None:
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

    for node in list(nodes):
        nodes.remove(node)

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(env_exr)
    try:
        env_tex.image.colorspace_settings.name = "Raw"
    except TypeError:
        env_tex.image.colorspace_settings.name = "Non-Color"
    env_tex.location = (-600, 0)

    background_env = nodes.new(type="ShaderNodeBackground")
    background_env.inputs["Strength"].default_value = 1.0
    background_env.location = (-350, 0)

    world_output = nodes.new(type="ShaderNodeOutputWorld")
    world_output.location = (-50, 0)

    links.new(env_tex.outputs["Color"], background_env.inputs["Color"])
    links.new(background_env.outputs["Background"], world_output.inputs["Surface"])


def get_floor_material(only_shadow: bool) -> bpy.types.Material:
    mat_name = "__ShadowFloorMaterial" if only_shadow else "__WhiteFloorMaterial"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    if only_shadow:
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        light_path = nodes.new(type="ShaderNodeLightPath")
        mix_shader = nodes.new(type="ShaderNodeMixShader")
        output = nodes.new(type="ShaderNodeOutputMaterial")

        links.new(light_path.outputs["Is Shadow Ray"], mix_shader.inputs["Fac"])
        links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
        links.new(diffuse.outputs["BSDF"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        mat.blend_method = "BLEND"
        mat.shadow_method = "NONE"
    else:
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.2
        mix_shader = nodes.new(type="ShaderNodeMixShader")
        mix_shader.inputs["Fac"].default_value = 0.35
        output = nodes.new(type="ShaderNodeOutputMaterial")

        links.new(diffuse.outputs["BSDF"], mix_shader.inputs[1])
        links.new(emission.outputs["Emission"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        mat.blend_method = "OPAQUE"
        mat.shadow_method = "OPAQUE"

    return mat


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not os.path.exists(args.scene_mesh):
        raise FileNotFoundError(f"Scene mesh not found: {args.scene_mesh}")
    if not os.path.exists(args.env_exr):
        raise FileNotFoundError(f"Environment EXR not found: {args.env_exr}")
    if args.external_ground and not os.path.exists(args.external_ground):
        raise FileNotFoundError(f"External ground not found: {args.external_ground}")
    if args.only_shadow and not (args.add_white_floor or args.external_ground):
        raise ValueError("--only_shadow requires --add_white_floor or --external_ground")

    with suppress_native_output():
        bpy.ops.wm.read_factory_settings(use_empty=True)
    ensure_object_mode()

    center, scale, min_z = compute_reference_center_scale(args.scene_mesh)

    animated_objects = import_scene_generic(args.mesh_file)
    if not animated_objects:
        raise RuntimeError(f"No objects imported from {args.mesh_file}")

    anim_center = center.copy()
    anim_min_z = min_z

    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()

    try:
        anim_center, _, anim_min_z, _, _ = compute_bounds(animated_objects)
    except RuntimeError:
        pass

    if args.render_no_texture:
        apply_untextured_material(animated_objects)

    ground_center = anim_center
    ground_min_z = min(min_z, anim_min_z)

    if args.external_ground:
        add_external_ground(args.external_ground, ground_center, ground_min_z, scale, args.only_shadow)
    elif args.add_white_floor:
        add_white_floor(ground_center, ground_min_z, scale, args.only_shadow)

    setup_hdr_lighting(args.env_exr)
    configure_cycles_cuda()

    cam = find_or_create_camera(args)
    position_camera(cam, center, scale, args)
    configure_camera_fov(cam, args)

    save_debug_blend(args.mesh_file)

    if args.render_frame is not None:
        args.save_frames = True
        args.save_video = False

    if not args.save_video and not args.save_frames:
        raise ValueError("Specify --save-video and/or --save-frames to produce output")

    frame_range = determine_frame_range()
    if args.render_frame is not None:
        requested_frame = int(args.render_frame)
        if requested_frame > frame_range[1]:
            print(
                f"Warning: requested frame {requested_frame} exceeds last animated frame "
                f"{frame_range[1]}; clamping to {frame_range[1]}"
            )
            requested_frame = frame_range[1]
        frame_range = (requested_frame, requested_frame)

    base_dir = os.path.abspath(args.output)
    base_name = f"rendered_video_{args.elev}_{args.azim}"

    if args.save_frames:
        frames_dir = os.path.join(base_dir, base_name)
        configure_render_frames(args, frame_range, frames_dir)
        render_quiet(frames_dir)

    if args.save_video:
        video_path = os.path.join(base_dir, f"{base_name}.mp4")
        configure_render_video(args, frame_range, video_path)
        render_quiet(video_path)


def get_floor_material(only_shadow: bool) -> bpy.types.Material:
    mat_name = "__ShadowFloorMaterial" if only_shadow else "__WhiteFloorMaterial"
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    if only_shadow:
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        transparent.location = (-400, 150)

        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        diffuse.location = (-400, -150)

        light_path = nodes.new(type="ShaderNodeLightPath")
        light_path.location = (-600, 0)

        mix_shader = nodes.new(type="ShaderNodeMixShader")
        mix_shader.location = (-150, 0)

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (100, 0)

        links.new(light_path.outputs["Is Shadow Ray"], mix_shader.inputs["Fac"])
        links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
        links.new(diffuse.outputs["BSDF"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        mat.blend_method = "BLEND"
        mat.shadow_method = "NONE"
    else:
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        diffuse.location = (-400, 150)

        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs["Strength"].default_value = 1.2
        emission.location = (-400, -150)

        mix_shader = nodes.new(type="ShaderNodeMixShader")
        mix_shader.inputs["Fac"].default_value = 0.35
        mix_shader.location = (-150, 0)

        output = nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (100, 0)

        links.new(diffuse.outputs["BSDF"], mix_shader.inputs[1])
        links.new(emission.outputs["Emission"], mix_shader.inputs[2])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        mat.blend_method = "OPAQUE"
        mat.shadow_method = "OPAQUE"

    return mat


def add_white_floor(center: mathutils.Vector, min_z: float, scale: float, only_shadow: bool) -> None:
    plane_size = max(4.0 / scale, 2.0)
    plane_height = min_z - 1e-3
    bpy.ops.mesh.primitive_plane_add(size=plane_size, enter_editmode=False, align="WORLD")
    floor = bpy.context.active_object
    floor.name = "ShadowFloor"
    floor.location = (center.x, center.y, plane_height)
    mat = get_floor_material(only_shadow)
    floor.data.materials.clear()
    floor.data.materials.append(mat)
    if hasattr(floor, "cycles"):
        floor.cycles.is_shadow_catcher = False if only_shadow else False


def add_external_ground(
    ground_path: str,
    center: mathutils.Vector,
    min_z: float,
    scale: float,
    only_shadow: bool,
) -> None:
    if not os.path.exists(ground_path):
        raise FileNotFoundError(f"External ground not found: {ground_path}")
    ground_objects = import_scene_generic(ground_path)
    ground_center, _, _, vmin, vmax = compute_bounds(ground_objects)
    extent = vmax - vmin
    max_extent = max(extent.x, extent.y, extent.z, 1e-6)
    floor_size = max(4.0 / scale, 2.0)
    overall_scale = floor_size / max_extent

    plane_height = min_z - 1e-3
    relative_min = (vmin.z - ground_center.z) * overall_scale
    translation = mathutils.Vector((center.x, center.y, plane_height - relative_min))

    def world_min_z(mesh_obj: bpy.types.Object) -> float:
        min_val = float("inf")
        matrix = mesh_obj.matrix_world.copy()
        data = mesh_obj.data
        if not data or data.is_editmode:
            return plane_height
        for vert in data.vertices:
            world_z = (matrix @ vert.co).z
            if world_z < min_val:
                min_val = world_z
        return min_val if min_val < float("inf") else plane_height

    bpy.context.view_layer.update()
    center_mat = mathutils.Matrix.Translation(-ground_center)
    scale_mat = mathutils.Matrix.Scale(overall_scale, 4)
    offset_mat = mathutils.Matrix.Translation((0.0, 0.0, -relative_min))
    translation_mat = mathutils.Matrix.Translation((center.x, center.y, plane_height))

    shadow_material = get_floor_material(True) if only_shadow else None

    mesh_mins = []
    for obj in ground_objects:
        obj.matrix_world = translation_mat @ offset_mat @ scale_mat @ center_mat @ obj.matrix_world
        if obj.type == "MESH" and hasattr(obj, "cycles"):
            obj.cycles.is_shadow_catcher = False if only_shadow else False
            if shadow_material is not None:
                obj.data.materials.clear()
                obj.data.materials.append(shadow_material)
        if obj.type == "MESH":
            obj_min_z = world_min_z(obj)
            mesh_mins.append((obj, obj_min_z))

    if mesh_mins:
        min_found = min(val for _, val in mesh_mins)
        delta = plane_height - min_found
        if abs(delta) > 1e-6:
            for obj in ground_objects:
                obj.location.z += delta
            bpy.context.view_layer.update()
            mesh_mins = [(obj, world_min_z(obj)) for obj, _ in mesh_mins]


if __name__ == "__main__":
    user_args = sys.argv[1:]
    main(user_args)
