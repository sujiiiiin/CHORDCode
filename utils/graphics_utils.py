import math
from typing import NamedTuple

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def convert_fov_to_opencv(fov_cameras: FoVPerspectiveCameras, image_wh):
    device = fov_cameras.device
    fov = fov_cameras.fov
    aspect_ratio = fov_cameras.aspect_ratio

    fov_rad = torch.deg2rad(fov)
    focal_length = 1.0 / torch.tan(fov_rad / 2)
    fx = focal_length
    fy = focal_length / aspect_ratio
    px = torch.zeros_like(fx)
    py = torch.zeros_like(fy)

    perspective_cameras = PerspectiveCameras(
        focal_length=torch.stack([fx, fy], dim=-1),
        principal_point=torch.stack([px, py], dim=-1),
        R=fov_cameras.R,
        T=fov_cameras.T,
        device=device,
    )
    R, T, K = opencv_from_cameras_projection(perspective_cameras, image_wh)
    return getWorld2View_batch(R, T), K


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = (C2W[:3, 3] + translate) * scale
    C2W[:3, 3] = cam_center
    return np.float32(np.linalg.inv(C2W))


def getWorld2View_batch(R, T):
    N = R.shape[0]
    device = R.device
    flip = torch.diag(torch.tensor([-1.0, -1.0, -1.0], device=device)).unsqueeze(0)
    R_cv = torch.bmm(flip.repeat(N, 1, 1), R)
    W2C = torch.eye(4, device=device).repeat(N, 1, 1)
    W2C[:, :3, :3] = R_cv
    W2C[:, :3, 3] = T
    return W2C


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tan_half_fovy = math.tan(fovY / 2)
    tan_half_fovx = math.tan(fovX / 2)
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
