import os

import numpy as np

from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=None,
        points=None,
        rgb=None,
        init=True,
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        _ = shuffle, resolution_scales

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.cameras_extent = 1.0

        if not init:
            return

        if self.loaded_iter is not None:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                )
            )
            return

        if points is None or rgb is None:
            raise ValueError(
                f"No saved point cloud found under '{self.model_path}/point_cloud' "
                "and no initialization points/rgb were provided."
            )

        pcd = BasicPointCloud(
            points=points,
            colors=SH2RGB(rgb),
            normals=np.zeros((points.shape[0], 3)),
        )
        self.gaussians.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration, save_ply=False):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        if save_ply:
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            return
        self.gaussians.save_pth(point_cloud_path)
