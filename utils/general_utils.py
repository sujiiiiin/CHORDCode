import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0]
        + r[:, 1] * r[:, 1]
        + r[:, 2] * r[:, 2]
        + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qw * qz)
    R[:, 0, 2] = 2 * (qx * qz + qw * qy)
    R[:, 1, 0] = 2 * (qx * qy + qw * qz)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qw * qx)
    R[:, 2, 0] = 2 * (qx * qz - qw * qy)
    R[:, 2, 1] = 2 * (qy * qz + qw * qx)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def safe_state(silent):
    old_f = sys.stdout

    class _StdoutProxy:
        def __init__(self, is_silent):
            self.silent = is_silent

        def write(self, x):
            if self.silent:
                return
            if x.endswith("\n"):
                old_f.write(
                    x.replace(
                        "\n",
                        " [{}]\n".format(datetime.now().strftime("%d/%m %H:%M:%S")),
                    )
                )
            else:
                old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = _StdoutProxy(silent)


def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def sample_from_segments(segments):
    disjoint = merge_intervals(segments)
    lengths = [r - l for l, r in disjoint]
    total_length = sum(lengths)
    pick = random.uniform(0, total_length)

    acc = 0
    for (l, r), length in zip(disjoint, lengths):
        acc += length
        if pick <= acc:
            return random.uniform(l, r)
