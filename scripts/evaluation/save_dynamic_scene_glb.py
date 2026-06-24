import copy
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from argparse import ArgumentParser
from pygltflib import (
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Buffer,
    BufferView,
    GLTF2,
    Node,
    Scene as GLTFScene,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, SDSOptimizationParams
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.scene_object_utils import resolve_scene_obj_num


warnings.filterwarnings("ignore")


_COMPONENT_TYPE_TO_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _node_local_matrix(node: Node) -> np.ndarray:
    if node.matrix is not None and len(node.matrix) == 16:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4)

    translation = np.array(
        node.translation if node.translation else [0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    rotation = np.array(
        node.rotation if node.rotation else [0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    scale = np.array(node.scale if node.scale else [1.0, 1.0, 1.0], dtype=np.float32)

    rot_mat = _quaternion_to_matrix(rotation)
    rot_mat = rot_mat * scale[None, :]

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rot_mat
    matrix[:3, 3] = translation
    return matrix


def _compute_mesh_world_transforms(gltf: GLTF2) -> Dict[int, List[np.ndarray]]:
    nodes = gltf.nodes or []
    world_mats: List[np.ndarray] = [np.eye(4, dtype=np.float32) for _ in range(len(nodes))]

    def _traverse(node_idx: int, parent_world: np.ndarray) -> None:
        local = _node_local_matrix(nodes[node_idx])
        current = parent_world @ local
        world_mats[node_idx] = current
        for child_idx in nodes[node_idx].children or []:
            _traverse(child_idx, current)

    default_scene = gltf.scene if gltf.scene is not None else 0
    scene = gltf.scenes[default_scene] if gltf.scenes else None
    if scene and scene.nodes:
        for root_idx in scene.nodes:
            _traverse(root_idx, np.eye(4, dtype=np.float32))

    mesh_to_world: Dict[int, List[np.ndarray]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        if node.mesh is not None:
            mesh_to_world[node.mesh].append(world_mats[idx])
    return mesh_to_world


def _read_accessor_data(gltf: GLTF2, accessor_idx: int, binary_blob: bytes) -> np.ndarray:
    accessor = gltf.accessors[accessor_idx]
    if accessor.sparse is not None:
        raise NotImplementedError("Sparse accessors are not supported in dynamic mesh export")
    if accessor.bufferView is None:
        raise ValueError("Accessor without bufferView is not supported")

    component_dtype = _COMPONENT_TYPE_TO_DTYPE.get(accessor.componentType)
    if component_dtype is None:
        raise ValueError(f"Unsupported component type: {accessor.componentType}")

    num_components = _NUM_COMPONENTS.get(accessor.type)
    if num_components is None:
        raise ValueError(f"Unsupported accessor type: {accessor.type}")

    buffer_view = gltf.bufferViews[accessor.bufferView]
    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = buffer_view.byteStride or (np.dtype(component_dtype).itemsize * num_components)

    data = np.empty((accessor.count, num_components), dtype=component_dtype)
    for idx in range(accessor.count):
        start = byte_offset + idx * stride
        array = np.frombuffer(binary_blob, dtype=component_dtype, count=num_components, offset=start)
        data[idx] = array
    return data.astype(np.float32)


def _ensure_list(container: Optional[List]) -> List:
    return container if container is not None else []


class GLTFSceneBuilder:
    def __init__(self):
        self.gltf = GLTF2()
        self.gltf.asset = Asset(version="2.0")
        self.gltf.buffers = [Buffer(byteLength=0)]
        self.gltf.bufferViews = []
        self.gltf.accessors = []
        self.gltf.materials = []
        self.gltf.textures = []
        self.gltf.images = []
        self.gltf.samplers = []
        self.gltf.meshes = []
        self.gltf.nodes = []
        self.gltf.scenes = [GLTFScene(nodes=[])]
        self.gltf.animations = []
        self.buffer_data = bytearray()
        self.time_accessor_cache: Dict[Tuple[int, float], int] = {}
        self.mesh_to_nodes: defaultdict[int, List[int]] = defaultdict(list)

    def _align_buffer(self, alignment: int = 4) -> None:
        padding = (-len(self.buffer_data)) % alignment
        if padding:
            self.buffer_data.extend(b"\x00" * padding)

    def add_buffer_view(
        self,
        data: bytes,
        byte_stride: Optional[int] = None,
        target: Optional[int] = None,
    ) -> int:
        self._align_buffer()
        offset = len(self.buffer_data)
        self.buffer_data.extend(data)
        view = BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(data),
            byteStride=byte_stride,
            target=target,
        )
        self.gltf.bufferViews.append(view)
        return len(self.gltf.bufferViews) - 1

    def add_accessor_from_array(
        self,
        array: np.ndarray,
        type_str: str,
        component_type: int = 5126,
        normalized: bool = False,
        compute_min_max: bool = False,
    ) -> int:
        array = np.asarray(array)
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        if array.ndim == 1:
            count = array.shape[0]
            num_components = 1
        elif array.ndim == 2:
            count, num_components = array.shape
        else:
            raise ValueError("Accessor array must be 1D or 2D")

        buffer_view_idx = self.add_buffer_view(array.tobytes())
        accessor = Accessor(
            bufferView=buffer_view_idx,
            byteOffset=0,
            componentType=component_type,
            count=int(count),
            type=type_str,
            normalized=normalized,
        )
        if compute_min_max:
            if num_components == 1:
                accessor.min = [float(array.min())]
                accessor.max = [float(array.max())]
            else:
                accessor.min = array.min(axis=0).tolist()
                accessor.max = array.max(axis=0).tolist()
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1

    def append_gltf(self, src_gltf: GLTF2) -> Dict[str, List[int]]:
        blob = src_gltf.binary_blob()
        buffer_view_map: List[int] = []
        accessor_map: List[int] = []
        image_map: List[int] = []
        sampler_map: List[int] = []
        texture_map: List[int] = []
        material_map: List[int] = []
        mesh_map: List[int] = []
        skin_map: List[int] = []
        node_map: List[int] = []

        for bv in _ensure_list(src_gltf.bufferViews):
            start = bv.byteOffset or 0
            end = start + (bv.byteLength or 0)
            data = blob[start:end]
            new_idx = self.add_buffer_view(data, byte_stride=bv.byteStride, target=bv.target)
            buffer_view_map.append(new_idx)

        for accessor in _ensure_list(src_gltf.accessors):
            new_accessor = copy.deepcopy(accessor)
            if new_accessor.bufferView is not None:
                new_accessor.bufferView = buffer_view_map[new_accessor.bufferView]
            if new_accessor.sparse is not None:
                if new_accessor.sparse.indices is not None and new_accessor.sparse.indices.bufferView is not None:
                    new_accessor.sparse.indices.bufferView = buffer_view_map[new_accessor.sparse.indices.bufferView]
                if new_accessor.sparse.values is not None and new_accessor.sparse.values.bufferView is not None:
                    new_accessor.sparse.values.bufferView = buffer_view_map[new_accessor.sparse.values.bufferView]
            self.gltf.accessors.append(new_accessor)
            accessor_map.append(len(self.gltf.accessors) - 1)

        for image in _ensure_list(src_gltf.images):
            new_image = copy.deepcopy(image)
            if new_image.bufferView is not None:
                new_image.bufferView = buffer_view_map[new_image.bufferView]
            self.gltf.images.append(new_image)
            image_map.append(len(self.gltf.images) - 1)

        for sampler in _ensure_list(src_gltf.samplers):
            self.gltf.samplers.append(copy.deepcopy(sampler))
            sampler_map.append(len(self.gltf.samplers) - 1)

        for texture in _ensure_list(src_gltf.textures):
            new_texture = copy.deepcopy(texture)
            if new_texture.sampler is not None:
                new_texture.sampler = sampler_map[new_texture.sampler]
            if new_texture.source is not None:
                new_texture.source = image_map[new_texture.source]
            self.gltf.textures.append(new_texture)
            texture_map.append(len(self.gltf.textures) - 1)

        for material in _ensure_list(src_gltf.materials):
            new_material = copy.deepcopy(material)
            if new_material.pbrMetallicRoughness is not None:
                pbr = new_material.pbrMetallicRoughness
                if pbr.baseColorTexture is not None and pbr.baseColorTexture.index is not None:
                    pbr.baseColorTexture.index = texture_map[pbr.baseColorTexture.index]
                if pbr.metallicRoughnessTexture is not None and pbr.metallicRoughnessTexture.index is not None:
                    pbr.metallicRoughnessTexture.index = texture_map[pbr.metallicRoughnessTexture.index]
            if new_material.extensions is not None:
                specular = new_material.extensions.get("KHR_materials_specular")
                if isinstance(specular, dict):
                    tex_info = specular.get("specularColorTexture")
                    if isinstance(tex_info, dict) and tex_info.get("index") is not None:
                        tex_info["index"] = texture_map[tex_info["index"]]
                    tex_info = specular.get("specularTexture")
                    if isinstance(tex_info, dict) and tex_info.get("index") is not None:
                        tex_info["index"] = texture_map[tex_info["index"]]
            if new_material.normalTexture is not None and new_material.normalTexture.index is not None:
                new_material.normalTexture.index = texture_map[new_material.normalTexture.index]
            if new_material.occlusionTexture is not None and new_material.occlusionTexture.index is not None:
                new_material.occlusionTexture.index = texture_map[new_material.occlusionTexture.index]
            if new_material.emissiveTexture is not None and new_material.emissiveTexture.index is not None:
                new_material.emissiveTexture.index = texture_map[new_material.emissiveTexture.index]
            self.gltf.materials.append(new_material)
            material_map.append(len(self.gltf.materials) - 1)

        for mesh in _ensure_list(src_gltf.meshes):
            new_mesh = copy.deepcopy(mesh)
            for primitive in new_mesh.primitives:
                attr_dict = primitive.attributes.__dict__
                for key, value in attr_dict.items():
                    if value is not None:
                        attr_dict[key] = accessor_map[value]
                if primitive.indices is not None:
                    primitive.indices = accessor_map[primitive.indices]
                if primitive.material is not None:
                    primitive.material = material_map[primitive.material]
                if primitive.targets:
                    new_targets = []
                    for target in primitive.targets:
                        new_target = {}
                        for attr, idx in target.items():
                            new_target[attr] = accessor_map[idx]
                        new_targets.append(new_target)
                    primitive.targets = new_targets
            self.gltf.meshes.append(new_mesh)
            mesh_map.append(len(self.gltf.meshes) - 1)

        skin_joint_refs = []
        for skin in _ensure_list(src_gltf.skins):
            new_skin = copy.deepcopy(skin)
            skeleton_ref = skin.skeleton
            joints = list(skin.joints) if skin.joints else []
            new_skin.joints = [0] * len(joints)
            if new_skin.inverseBindMatrices is not None:
                new_skin.inverseBindMatrices = accessor_map[new_skin.inverseBindMatrices]
            new_skin.skeleton = None if skeleton_ref is None else skeleton_ref
            self.gltf.skins.append(new_skin)
            skin_map.append(len(self.gltf.skins) - 1)
            skin_joint_refs.append((len(self.gltf.skins) - 1, joints, skeleton_ref))

        node_offset = len(self.gltf.nodes)
        for node in _ensure_list(src_gltf.nodes):
            new_node = copy.deepcopy(node)
            if new_node.mesh is not None:
                new_node.mesh = mesh_map[new_node.mesh]
            if new_node.children:
                new_node.children = [child + node_offset for child in new_node.children]
            if new_node.skin is not None:
                new_node.skin = skin_map[new_node.skin]
            self.gltf.nodes.append(new_node)
            node_map.append(len(self.gltf.nodes) - 1)

        for skin_idx, joints, skeleton_ref in skin_joint_refs:
            mapped_joints = [node_map[j] for j in joints]
            self.gltf.skins[skin_idx].joints = mapped_joints
            if skeleton_ref is not None:
                self.gltf.skins[skin_idx].skeleton = node_map[skeleton_ref]

        for new_idx in node_map:
            node = self.gltf.nodes[new_idx]
            if node.mesh is not None:
                self.mesh_to_nodes[node.mesh].append(new_idx)

        src_children = set()
        for node in _ensure_list(src_gltf.nodes):
            if node.children:
                src_children.update(node.children)
        root_nodes = [node_map[idx] for idx in range(len(node_map)) if idx not in src_children]
        self.gltf.scenes[0].nodes.extend(root_nodes)

        return {
            "bufferViews": buffer_view_map,
            "accessors": accessor_map,
            "images": image_map,
            "samplers": sampler_map,
            "textures": texture_map,
            "materials": material_map,
            "meshes": mesh_map,
            "nodes": node_map,
            "root_nodes": root_nodes,
        }

    def _time_accessor(self, num_frames: int, fps: float) -> int:
        fps = max(float(fps), 1e-6)
        key = (num_frames, fps)
        if key in self.time_accessor_cache:
            return self.time_accessor_cache[key]
        times = np.arange(num_frames, dtype=np.float32) / fps
        accessor_idx = self.add_accessor_from_array(times, type_str="SCALAR", compute_min_max=True)
        self.time_accessor_cache[key] = accessor_idx
        return accessor_idx

    def add_weights_animation(
        self,
        node_indices: List[int],
        num_frames: int,
        num_targets: int,
        fps: float,
    ) -> None:
        if num_targets <= 0 or not node_indices:
            return
        time_accessor = self._time_accessor(num_frames, fps)
        weights = np.zeros((num_frames, num_targets), dtype=np.float32)
        for frame_idx in range(1, num_frames):
            weights[frame_idx, frame_idx - 1] = 1.0

        weights_accessor = self.add_accessor_from_array(
            weights.reshape(-1),
            type_str="SCALAR",
            compute_min_max=False,
        )

        if not self.gltf.animations:
            self.gltf.animations = [Animation(name="Deformation", channels=[], samplers=[])]
        animation = self.gltf.animations[0]

        sampler_idx = len(animation.samplers)
        animation.samplers.append(
            AnimationSampler(input=time_accessor, output=weights_accessor, interpolation="STEP")
        )

        for node_idx in node_indices:
            animation.channels.append(
                AnimationChannel(
                    sampler=sampler_idx,
                    target=AnimationChannelTarget(node=node_idx, path="weights"),
                )
            )

    def finalize(self) -> None:
        self._align_buffer()
        self.gltf.buffers[0].byteLength = len(self.buffer_data)
        self.gltf.set_binary_blob(bytes(self.buffer_data))


def _normalize_exclude_objs(exclude_objs: Optional[List[str]]) -> set[str]:
    normalized: set[str] = set()
    for value in exclude_objs or []:
        value = str(value).strip()
        if not value:
            continue
        if value.startswith("obj_"):
            normalized.add(value)
        else:
            normalized.add(f"obj_{value}")
    return normalized


def _load_dynamic_gaussian_checkpoint(opt, gaussians, scene, ckpt_path):
    from scene.dynamic_gaussian_model import DynamicGaussianModel

    dynamic_gaussians = DynamicGaussianModel(
        gaussians,
        opt.frame_num,
        scene.cameras_extent,
        n_cp_num=opt.n_cp_num,
    )
    dynamic_gaussians.load_pth(ckpt_path)
    return dynamic_gaussians


def collect_mesh_frames(
    src_gltf: GLTF2,
    dynamic_gaussians,
    mesh_world_transforms: Dict[int, List[np.ndarray]],
    scene_center: np.ndarray,
    scene_scale: float,
    current_frame_limit: Optional[int] = None,
) -> Tuple[
    Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    int,
    int,
]:
    primitives_keys: List[Tuple[int, int, int]] = []
    world_vertex_blocks: List[np.ndarray] = []
    inverse_transforms: List[np.ndarray] = []
    local_vertices: List[np.ndarray] = []
    blob = src_gltf.binary_blob()

    if not src_gltf.meshes:
        return {}, 0, 0

    for mesh_idx, mesh in enumerate(src_gltf.meshes):
        world_mats = mesh_world_transforms.get(mesh_idx, [np.eye(4, dtype=np.float32)])
        for prim_idx, primitive in enumerate(mesh.primitives):
            if primitive.attributes is None or primitive.attributes.POSITION is None:
                continue
            positions = _read_accessor_data(src_gltf, primitive.attributes.POSITION, blob)
            world_mat = world_mats[0]
            rot = world_mat[:3, :3]
            trans = world_mat[:3, 3]
            world_positions = positions @ rot.T + trans
            primitives_keys.append((mesh_idx, prim_idx, 0))
            world_vertex_blocks.append(world_positions)
            inverse_transforms.append(np.linalg.inv(world_mat))
            local_vertices.append(positions.copy())

    if not world_vertex_blocks:
        return {}, 0, 0

    device: Optional[torch.device] = None
    if dynamic_gaussians is not None:
        if hasattr(dynamic_gaussians, "cp_center"):
            device = dynamic_gaussians.cp_center.device
        elif hasattr(dynamic_gaussians, "gaussians") and hasattr(dynamic_gaussians.gaussians, "_xyz"):
            device = dynamic_gaussians.gaussians._xyz.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    concat_vertices_np = np.concatenate(world_vertex_blocks, axis=0).astype(np.float32)
    world_vertices = torch.from_numpy(concat_vertices_np).to(device)
    center = torch.as_tensor(scene_center.astype(np.float32), dtype=torch.float32, device=device)
    scale = torch.tensor(float(scene_scale), dtype=torch.float32, device=device)
    processed_tensor = (world_vertices - center) * scale

    with torch.no_grad():
        sequence_tensors: List[torch.Tensor] = []
        current_frames_appended = 0

        def _gather_frames(gaussian, start_idx: int, end_idx: int):
            if gaussian is None or end_idx <= start_idx:
                return None
            if hasattr(gaussian, "query_xyz_time"):
                frames: List[torch.Tensor] = []
                total_time = getattr(gaussian, "total_time", end_idx)
                actual_end = min(end_idx, total_time)
                if actual_end <= start_idx:
                    return None
                for frame_idx in range(start_idx, actual_end):
                    frame_vertices = gaussian.query_xyz_time(
                        processed_tensor,
                        frame_idx,
                        detach_node_radius=True,
                    )
                    frames.append(frame_vertices.detach())
                if not frames:
                    return None
                return torch.stack(frames, dim=0)
            if hasattr(gaussian, "query_xyz_time_whole"):
                frame_tensor = gaussian.query_xyz_time_whole(
                    processed_tensor,
                    detach_node_radius=True,
                )
                actual_end = min(end_idx, frame_tensor.shape[0])
                if actual_end <= start_idx:
                    return None
                return frame_tensor[start_idx:actual_end].detach()
            raise TypeError("Dynamic object does not support vertex deformation queries")

        if dynamic_gaussians is not None:
            total_time = getattr(dynamic_gaussians, "total_time", 1)
            start_idx = 0
            end_idx = total_time
            if current_frame_limit is not None:
                end_idx = min(total_time, start_idx + current_frame_limit)
            tensor = _gather_frames(dynamic_gaussians, start_idx, end_idx)
            if tensor is not None and tensor.numel() > 0:
                sequence_tensors.append(tensor)
                current_frames_appended = tensor.shape[0]

        if not sequence_tensors:
            return {}, 0, current_frames_appended

        deformed = torch.cat(sequence_tensors, dim=0)
        total_frames_appended = deformed.shape[0]

    frame_dict: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    counts = [block.shape[0] for block in world_vertex_blocks]
    offsets = np.cumsum([0] + counts)

    scale_value = float(scale.cpu().item())
    center_np = center.cpu().numpy()
    deformed_np = deformed.cpu().numpy().astype(np.float32)
    deformed_np = deformed_np / scale_value + center_np

    for idx, (key, start, end) in enumerate(zip(primitives_keys, offsets[:-1], offsets[1:])):
        inv_mat = inverse_transforms[idx]
        world_positions = deformed_np[:, start:end, :]
        hom = np.concatenate(
            [
                world_positions,
                np.ones((total_frames_appended, world_positions.shape[1], 1), dtype=np.float32),
            ],
            axis=-1,
        )
        local_positions = hom @ inv_mat.T
        local_positions = local_positions[..., :3]
        base_positions = local_vertices[idx].astype(np.float32)
        key_simple = key[:2]
        if key_simple in frame_dict:
            continue
        frame_dict[key_simple] = (local_positions, base_positions)

    return frame_dict, total_frames_appended, current_frames_appended


def apply_dynamic_to_builder(
    builder: GLTFSceneBuilder,
    frame_data: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    mesh_map: List[int],
    fps: float,
) -> None:
    if not frame_data:
        return

    num_frames = None
    targets_per_mesh: Dict[int, int] = {}

    for (src_mesh_idx, prim_idx), (frames, base_positions) in frame_data.items():
        new_mesh_idx = mesh_map[src_mesh_idx]
        mesh = builder.gltf.meshes[new_mesh_idx]
        primitive = mesh.primitives[prim_idx]

        base_positions = base_positions.astype(np.float32)
        new_accessor = builder.add_accessor_from_array(base_positions, type_str="VEC3", compute_min_max=True)
        primitive.attributes.POSITION = new_accessor

        if num_frames is None:
            num_frames = frames.shape[0]
        targets = []
        for frame_id in range(1, frames.shape[0]):
            delta = (frames[frame_id] - base_positions).astype(np.float32)
            accessor_idx = builder.add_accessor_from_array(delta, type_str="VEC3", compute_min_max=False)
            targets.append({"POSITION": accessor_idx})
        primitive.targets = targets
        targets_per_mesh.setdefault(new_mesh_idx, len(targets))

    if num_frames is None:
        return

    fps = max(float(fps), 1.0)
    for mesh_idx, target_count in targets_per_mesh.items():
        if target_count <= 0:
            continue
        mesh = builder.gltf.meshes[mesh_idx]
        mesh.weights = [0.0 for _ in range(target_count)]
        node_indices = builder.mesh_to_nodes.get(mesh_idx, [])
        builder.add_weights_animation(node_indices, num_frames, target_count, fps)


def export_dynamic_scene(
    dataset,
    opt: SDSOptimizationParams,
    checkpoint: int,
    lst_frame: Optional[int] = None,
    exclude_objs: Optional[List[str]] = None,
    exclude_plane: bool = False,
):
    builder = GLTFSceneBuilder()
    excluded = _normalize_exclude_objs(exclude_objs)

    scene_mesh_path = os.path.join(dataset.mesh_source_path, "scene.glb")
    if not os.path.exists(scene_mesh_path):
        raise FileNotFoundError(f"Scene mesh not found at {scene_mesh_path}")

    scene_gltf = GLTF2().load(scene_mesh_path)
    scene_transforms = _compute_mesh_world_transforms(scene_gltf)

    scene_vertices: List[np.ndarray] = []
    scene_blob = scene_gltf.binary_blob()
    for mesh_idx, mesh in enumerate(scene_gltf.meshes or []):
        world_mats = scene_transforms.get(mesh_idx, [np.eye(4, dtype=np.float32)])
        for primitive in mesh.primitives:
            if primitive.attributes is None or primitive.attributes.POSITION is None:
                continue
            positions = _read_accessor_data(scene_gltf, primitive.attributes.POSITION, scene_blob)
            for world_mat in world_mats:
                rot = world_mat[:3, :3]
                trans = world_mat[:3, 3]
                scene_vertices.append(positions @ rot.T + trans)

    if not scene_vertices:
        raise ValueError("Scene mesh contains no vertex data for scaling reference")

    scene_concat = np.concatenate(scene_vertices, axis=0)
    scene_vmin = scene_concat.min(axis=0)
    scene_vmax = scene_concat.max(axis=0)
    scene_center = (scene_vmin + scene_vmax) * 0.5
    scene_extent = np.maximum(scene_vmax - scene_vmin, 1e-9)
    scene_scale = 1.2 / np.max(scene_extent)

    for obj_idx in range(opt.obj_num):
        obj_name = f"obj_{obj_idx}"
        if obj_name in excluded:
            print(f"Skipping excluded object {obj_name}")
            continue

        cur_dataset = copy.deepcopy(dataset)
        cur_dataset.model_path = os.path.join(dataset.model_path, obj_name)

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(cur_dataset, gaussians, load_iteration=-1)
        ckpt_path = os.path.join(
            dataset.model_path,
            opt.ex_name,
            "deform",
            f"deform_{checkpoint}",
            f"{obj_name}.pth",
        )

        dynamic_gaussians = None
        is_dynamic = obj_idx != opt.static_id

        if is_dynamic and os.path.exists(ckpt_path):
            dynamic_gaussians = _load_dynamic_gaussian_checkpoint(opt, gaussians, scene, ckpt_path)
        elif is_dynamic:
            raise FileNotFoundError(f"Dynamic checkpoint not found for {obj_name}: {ckpt_path}")
        else:
            print(f"{obj_name} is treated as static in this release exporter.")

        base_mesh_path = os.path.join(dataset.mesh_source_path, f"{obj_name}.glb")
        mesh_path = base_mesh_path
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found for {obj_name}: {mesh_path}")
        if exclude_plane and obj_idx == opt.static_id:
            static_obj_path = os.path.join(dataset.mesh_source_path, "static_obj.glb")
            if os.path.exists(static_obj_path):
                mesh_path = static_obj_path

        src_gltf = GLTF2().load(mesh_path)
        mapping = builder.append_gltf(src_gltf)

        if dynamic_gaussians is not None:
            mesh_world_transforms = _compute_mesh_world_transforms(src_gltf)
            frame_data, total_frames, current_seq_frames = collect_mesh_frames(
                src_gltf,
                dynamic_gaussians,
                mesh_world_transforms,
                scene_center,
                scene_scale,
                current_frame_limit=lst_frame,
            )
            if lst_frame is not None:
                print(f"{obj_name}: current frames={current_seq_frames}, total={total_frames}")
            if total_frames > 0:
                apply_dynamic_to_builder(
                    builder,
                    frame_data,
                    mapping["meshes"],
                    fps=opt.save_fps if opt.save_fps > 0 else 12.0,
                )

        del scene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dataset.model_path = os.path.join(dataset.model_path, opt.ex_name)
    os.makedirs(dataset.model_path, exist_ok=True)
    output_dir = os.path.join(dataset.model_path, "saved_meshes", f"iteration_{checkpoint}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dynamic_scene.glb")

    builder.finalize()
    builder.gltf.save(output_path)
    print(f"Dynamic scene saved to {output_path}")


def main():
    parser = ArgumentParser(description="Export dynamic scene as animated glTF")
    lp = ModelParams(parser)
    op = SDSOptimizationParams(parser)

    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint iteration to load")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--lst_frame", type=int, default=None, help="Limit exported frames from current scene")
    parser.add_argument(
        "--exclude_objs",
        nargs="+",
        type=str,
        default=[],
        help="Objects to exclude, e.g. '0 2' or 'obj_0 obj_2'",
    )
    parser.add_argument(
        "--exclude_plane",
        action="store_true",
        default=False,
        help="Use static_obj.glb for the static background when available",
    )

    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    dataset = lp.extract(args)
    opt = op.extract(args)
    resolve_scene_obj_num(opt, dataset.mesh_source_path)

    if args.checkpoint < 0:
        raise ValueError("Please provide a valid --checkpoint")

    export_dynamic_scene(
        dataset,
        opt,
        args.checkpoint,
        lst_frame=args.lst_frame,
        exclude_objs=args.exclude_objs,
        exclude_plane=args.exclude_plane,
    )


if __name__ == "__main__":
    main()
