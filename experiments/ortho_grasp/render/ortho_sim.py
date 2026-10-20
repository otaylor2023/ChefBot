"""V4: realistic orthographic renders of the cup (spec §6.2).

MuJoCo's free camera in orthographic mode (no model edits needed), framed by
the plot spec: looking down (top) or along robot +y (side), image right =
robot +x. Everything but the cup is whited out (the side view's counter is the
plot's own counter band). `check_alignment` projects mesh points through the
plot spec and fails loudly if they miss the cup by more than 2 px.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from experiments.ortho_grasp.frames import to_world
from experiments.ortho_grasp.render.plots import PlotSpec

ALIGN_TOLERANCE_PX = 2
CAMERA_DISTANCE = 0.30  # m from the view's center


class AlignmentError(RuntimeError):
    pass


def camera(world, spec: PlotSpec, depth_center: float) -> mujoco.MjvCamera:
    """`depth_center`: the robot-frame coordinate along the view direction to center on (z or y)."""
    h_mid, v_mid = (spec.h_range[0] + spec.h_range[1]) / 2, (spec.v_range[0] + spec.v_range[1]) / 2
    if spec.view == "top":
        center, elevation = np.array([h_mid, v_mid, depth_center]), -90.0
    else:
        center, elevation = np.array([h_mid, depth_center, v_mid]), 0.0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.orthographic = 1
    cam.lookat[:] = to_world(center[None], world.base)[0]
    cam.distance = CAMERA_DISTANCE  # close: the wall cabinets above and anything beside it are behind the camera
    cam.elevation = elevation
    cam.azimuth = math.degrees(world.base.yaw) + 90.0  # looking along robot +y (side); image up = robot +y (top)
    return cam


def render(world, spec: PlotSpec, cup_root: int, depth_center: float) -> tuple[np.ndarray, np.ndarray]:
    """(RGBA area x area, transparent but for the cup; cup mask)."""
    m = world.env.model
    cam = camera(world, spec, depth_center)
    fovy, ortho = float(m.vis.global_.fovy), int(m.vis.global_.orthographic)
    # The free camera's projection comes from the model's global visual settings:
    # orthographic, with fovy the view's height in meters. Restored after.
    m.vis.global_.fovy, m.vis.global_.orthographic = spec.h_range[1] - spec.h_range[0], 1
    try:
        with world.isolated(cup_root):  # distractors parked for the render: they could stand in front of the cup
            rgb = world.render(cam, spec.area, spec.area)
            seg = world.render(cam, spec.area, spec.area, segmentation=True)
    finally:
        m.vis.global_.fovy, m.vis.global_.orthographic = fovy, ortho
    mask = world.segment_roots(seg) == cup_root
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)  # RGBA: transparent where it is not the cup
    out[mask, :3], out[mask, 3] = rgb[mask], 255
    return out, mask


def check_alignment(spec: PlotSpec, mask: np.ndarray, cup_points: np.ndarray, rng: np.random.Generator, n: int = 10) -> float:
    """Project `n` visible cup points through the spec; every one must land within
    ALIGN_TOLERANCE_PX of a cup pixel in `mask`. Returns the worst miss (px)."""
    if spec.view == "top":
        keep = cup_points[:, 2] > np.percentile(cup_points[:, 2], 90)  # the rim: seen from above
    else:
        keep = cup_points[:, 1] < np.percentile(cup_points[:, 1], 10)  # the near side: seen along +y
    pts = cup_points[keep]
    pts = pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)]
    uv = spec.to_pixel(spec.project(pts)) - [spec.left, spec.top]
    ys, xs = np.nonzero(mask)
    cup_px = np.stack([xs + 0.5, ys + 0.5], axis=1)
    worst = 0.0
    for p in uv:
        miss = float(np.min(np.linalg.norm(cup_px - p, axis=1))) if len(cup_px) else np.inf
        worst = max(worst, miss)
    if worst > ALIGN_TOLERANCE_PX:
        raise AlignmentError(f"{spec.view} view: a cup point lands {worst:.1f} px from the rendered cup (> {ALIGN_TOLERANCE_PX})")
    return worst
