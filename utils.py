"""ZipNeRF/utils.py"""

import math

import numpy as np

from Cameras.Perspective import PerspectiveCamera


def next_multiple(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def pixel_radius(camera: PerspectiveCamera) -> float:
    """Returns the cone radius at unit distance (2 / sqrt(12) times the mean pixel size)."""
    return float(1 / camera.focal_x + 1 / camera.focal_y) / math.sqrt(12)


def ray_warp(near: float, far: float, p: float, premult: float) -> tuple[float, ...]:
    """Computes the constants of the mapping between normalized and metric ray distances in float32 (see RayWarp)."""
    def fn(x: float) -> np.float32:
        x = np.float32(x) * np.float32(premult)
        y = np.float32(abs(p - 1) / p) * ((np.abs(x) / np.float32(abs(p - 1)) + np.float32(1)) ** np.float32(p) - np.float32(1))
        return np.copysign(y, x)
    # one ulp below the asymptote, which bounds the far end of the ray
    y_max = np.nextafter(np.float32((p - 1) / p), np.float32(0)) if p < 0 else np.float32(np.inf)
    return near, far, float(fn(near)), float(fn(far)), p, premult, float(y_max)


def schlick_bias(x: float, s: float) -> float:
    """Computes Schlick's bias function (https://arxiv.org/abs/2010.09714)."""
    return (s * x) / ((s - 1) * x + 1)
