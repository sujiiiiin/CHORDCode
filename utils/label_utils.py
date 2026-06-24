import cv2
import numpy as np
import torch


def _build_hsv_palette(num: int) -> torch.Tensor:
    if num <= 0:
        return torch.empty((0, 3), dtype=torch.float32, device="cuda")

    hue = np.linspace(0.0, 1.0, num, endpoint=False, dtype=np.float32)
    sat_levels = np.array([0.95, 0.8, 0.65], dtype=np.float32)
    val_levels = np.array([1.0, 0.9], dtype=np.float32)

    hsv = np.zeros((num, 1, 3), dtype=np.float32)
    hsv[:, 0, 0] = hue * 179.0
    hsv[:, 0, 1] = sat_levels[np.arange(num) % len(sat_levels)] * 255.0
    hsv[:, 0, 2] = val_levels[(np.arange(num) // len(sat_levels)) % len(val_levels)] * 255.0

    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(num, 3)
    return torch.from_numpy(rgb).float().cuda() / 255.0


def num2rgb(num: int) -> torch.Tensor:
    return _build_hsv_palette(num)
