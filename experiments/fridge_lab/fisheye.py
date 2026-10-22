"""A fisheye camera for MuJoCo, which renders only pinhole views: render one wide, high-resolution pinhole view and
warp it into the fisheye image (OpenCV's fisheye model, the one a real calibration would give).

The lens is MessyMem's wrist camera, an Arducam IMX291 USB module with a 160 degree (diagonal) fisheye lens:
1920x1080, here rendered at 640x360 (about 140 x 78 degrees). Its distortion is a mild departure from the
equidistant model (theta_d = theta (1 + k1 theta^2 + k2 theta^4 + ...)).
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

W, H = 640, 360
DIAG_FOV_DEG = 160.0
D = (-0.02, 0.002, 0.0, 0.0)  # k1..k4
PIXELS_PER_TAN = 200.0  # density of the pinhole render at its center


def _theta_d(theta: float) -> float:
    return theta * (1 + D[0] * theta**2 + D[1] * theta**4 + D[2] * theta**6 + D[3] * theta**8)


def nominal_f() -> float:
    """Focal length (px) that puts the image corners at half the diagonal field of view."""
    return math.hypot(W / 2, H / 2) / _theta_d(math.radians(DIAG_FOV_DEG / 2))


def K_of(f: float, w: int = W, h: int = H) -> np.ndarray:
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def rays(f: float, scale: float = 1.0) -> np.ndarray:
    """Unit rays (OpenCV camera axes) of every pixel of the fisheye image, (h, w, 3)."""
    w, h = round(W * scale), round(H * scale)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.fisheye.undistortPoints(pts, K_of(f * scale, w, h), np.array(D)).reshape(h, w, 2)  # x/z, y/z
    r = np.concatenate([und, np.ones((h, w, 1))], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


@lru_cache(maxsize=8)
def pinhole_extent(f_max: float) -> tuple[float, float]:
    """Half-extents (tan x, tan y) a pinhole render must cover to contain the fisheye image at focal f_max (the widest)."""
    r = rays(f_max * 0.97, 0.25)  # a slightly wider lens than any episode's, sampled coarsely
    return float(np.abs(r[..., 0] / r[..., 2]).max() * 1.02), float(np.abs(r[..., 1] / r[..., 2]).max() * 1.02)


@lru_cache(maxsize=16)
def maps(f: float, scale: float, ext: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """cv2.remap maps from the pinhole render to the fisheye image (scaled by `scale`), and the render's size."""
    tx, ty = ext
    pw, ph = round(2 * tx * PIXELS_PER_TAN * scale), round(2 * ty * PIXELS_PER_TAN * scale)
    r = rays(f, scale)
    fp = PIXELS_PER_TAN * scale
    mx = (pw / 2 + fp * r[..., 0] / r[..., 2]).astype(np.float32)
    my = (ph / 2 + fp * r[..., 1] / r[..., 2]).astype(np.float32)
    return mx, my, pw, ph


def fovy_deg(ext: tuple[float, float]) -> float:
    """The pinhole render's vertical field of view."""
    return math.degrees(2 * math.atan(ext[1]))


def warp(pinhole: np.ndarray, f: float, scale: float, ext: tuple[float, float]) -> np.ndarray:
    mx, my, _, _ = maps(f, scale, ext)
    return cv2.remap(pinhole, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
