import re
from pathlib import Path


_OBJ_MESH_RE = re.compile(r"obj_(\d+)\.glb")


def discover_scene_object_ids(mesh_source_path):
    mesh_root = Path(mesh_source_path)
    if not mesh_root.is_dir():
        raise FileNotFoundError(
            f"mesh_source_path does not exist or is not a directory: {mesh_root}"
        )

    object_ids = []
    for path in mesh_root.iterdir():
        if not path.is_file():
            continue
        match = _OBJ_MESH_RE.fullmatch(path.name)
        if match:
            object_ids.append(int(match.group(1)))

    if not object_ids:
        raise FileNotFoundError(f"No top-level obj_<id>.glb files found under {mesh_root}")

    object_ids = sorted(object_ids)
    expected_ids = list(range(object_ids[-1] + 1))
    if object_ids != expected_ids:
        missing_ids = sorted(set(expected_ids) - set(object_ids))
        missing_names = ", ".join(f"obj_{obj_id}.glb" for obj_id in missing_ids)
        raise ValueError(
            "Scene object ids must be contiguous from obj_0.glb. "
            f"Missing: {missing_names}"
        )

    return object_ids


def detect_scene_obj_num(mesh_source_path):
    object_ids = discover_scene_object_ids(mesh_source_path)
    return object_ids[-1] + 1


def resolve_scene_obj_num(opt, mesh_source_path):
    if opt.obj_num == -1:
        opt.obj_num = detect_scene_obj_num(mesh_source_path)
    elif opt.obj_num <= 0:
        raise ValueError("--obj_num must be -1 for auto-detection or a positive integer.")

    if opt.static_id < 0 or opt.static_id >= opt.obj_num:
        raise ValueError(
            f"--static_id {opt.static_id} is outside the resolved object range "
            f"[0, {opt.obj_num - 1}]."
        )

    return opt.obj_num
