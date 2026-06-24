import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tan_half_fovy = math.tan(fovY / 2)
    tan_half_fovx = math.tan(fovX / 2)

    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tan_half_fovx
    P[1, 1] = 1 / tan_half_fovy
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        w2c_gl = np.linalg.inv(c2w).astype(np.float32)
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        w2c = flip @ w2c_gl

        self.world_view_transform = torch.from_numpy(w2c).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = torch.from_numpy(c2w[:3, 3].astype(np.float32)).cuda()


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    if not opengl:
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    return np.stack([right_vector, up_vector, forward_vector], axis=1)


def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = look_at(campos, target, opengl)
    pose[:3, 3] = campos
    return pose


def get_cam_pos_from_pose(elev, azim, cam_radius, *, is_degree=True, target=None):
    if any(torch.is_tensor(val) for val in (elev, azim, cam_radius)):
        device = next(
            (val.device for val in (elev, azim, cam_radius) if torch.is_tensor(val)),
            torch.device("cpu"),
        )
        dtype = next(
            (val.dtype for val in (elev, azim, cam_radius) if torch.is_tensor(val)),
            torch.float32,
        )

        def _to_tensor(value):
            if torch.is_tensor(value):
                return value.to(device=device, dtype=dtype)
            return torch.tensor(value, device=device, dtype=dtype)

        elev_t = _to_tensor(elev)
        azim_t = _to_tensor(azim)
        radius_t = _to_tensor(cam_radius)

        if is_degree:
            elev_t = elev_t * (math.pi / 180.0)
            azim_t = azim_t * (math.pi / 180.0)

        x = radius_t * torch.cos(elev_t) * torch.sin(azim_t)
        y = -radius_t * torch.sin(elev_t)
        z = radius_t * torch.cos(elev_t) * torch.cos(azim_t)
        campos = torch.stack((x, y, z), dim=-1)
        if target is not None:
            campos = campos + _to_tensor(target)
        return campos

    elev_f = math.radians(float(elev)) if is_degree else float(elev)
    azim_f = math.radians(float(azim)) if is_degree else float(azim)
    radius_f = float(cam_radius)
    campos = np.array(
        [
            radius_f * math.cos(elev_f) * math.sin(azim_f),
            -radius_f * math.sin(elev_f),
            radius_f * math.cos(elev_f) * math.cos(azim_f),
        ],
        dtype=np.float32,
    )
    if target is not None:
        campos = campos + np.asarray(target, dtype=np.float32)
    return campos


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r
        self.fovy = np.deg2rad(fovy)
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    @property
    def pose(self):
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = self.radius
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        pose = rot @ pose
        pose[:3, 3] -= self.center
        return pose

    @property
    def view(self):
        return np.linalg.inv(self.pose)

    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)

    def orbit(self, dx, dy):
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])
