import math

import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform


def get_pytorch3d_camera(R, T, FoVx, FoVy, image_width, image_height, device="cuda"):
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    fx = image_width / (2 * tanfovx)
    fy = image_height / (2 * tanfovy)

    focal_length = torch.tensor(
        [[2 * fx / image_width, 2 * fy / image_height]], device=device
    )
    principal_point = torch.tensor([[0.0, 0.0]], device=device)
    focal_length = focal_length.repeat(R.shape[0], 1)
    principal_point = principal_point.repeat(R.shape[0], 1)

    return PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point,
        device=device,
    )


def get_batched_cams(
    radius,
    lookat,
    num_views=20,
    elev_range=(0,),
    azim_range=(0, 360),
    device="cuda",
    rand_sample=False,
):
    lookat = (
        torch.as_tensor(lookat, dtype=torch.float32, device=device).unsqueeze(0)
        if isinstance(lookat, (list, np.ndarray))
        else lookat
    )
    radius = torch.as_tensor(radius, dtype=torch.float32, device=device)

    if not rand_sample:
        azim_vals = torch.linspace(azim_range[0], azim_range[1], num_views, device=device)
    else:
        azim_vals = (
            torch.rand(num_views, device=device) * (azim_range[1] - azim_range[0])
            + azim_range[0]
        )

    if len(elev_range) == 1:
        elev_vals = torch.full_like(azim_vals, elev_range[0])
    elif not rand_sample:
        elev_vals = torch.linspace(elev_range[0], elev_range[1], num_views, device=device)
    else:
        elev_vals = (
            torch.rand(num_views, device=device) * (elev_range[1] - elev_range[0])
            + elev_range[0]
        )

    R, T = look_at_view_transform(
        dist=radius,
        elev=elev_vals,
        azim=azim_vals,
        at=lookat.expand(num_views, -1),
        device=device,
    )
    return {"R": R, "T": T, "viewmat": getWorld2View2_Pt3d(R, T, device=device)}


def getWorld2View2_Pt3d(
    R,
    T,
    translate=torch.tensor([0.0, 0.0, 0.0]),
    scale=1.0,
    device="cuda",
):
    N = R.shape[0]
    translate = translate.to(device).unsqueeze(0).expand(N, -1)
    scale = torch.tensor(scale, dtype=torch.float32, device=device)

    W2C = torch.eye(4, device=device).repeat(N, 1, 1)
    W2C[:, :3, :3] = R
    W2C[:, :3, 3] = T

    C2W = torch.linalg.inv(W2C)
    cam_center = (C2W[:, :3, 3] + translate) * scale
    C2W[:, :3, 3] = cam_center

    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device)).unsqueeze(0)
    C2W[:, :3, :3] = torch.bmm(C2W[:, :3, :3], flip.repeat(N, 1, 1))
    return torch.linalg.inv(C2W).float()
