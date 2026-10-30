"""Handle detection (sim stand-in for a part segmenter) and where to hold a handle."""

import numpy as np
import pytest

from chefbot.environments.robocasa.handles import detect_handle
from chefbot.grasping.handle import GRIP_FROM_BODY, handle_grasps

FLOOR = 0.9


def pan(handle_length=0.20, handle_gap=(0.06, 0.13), yaw=0.0):
    """A pan as a sparse point cloud: a 22 cm disk centered at x = -0.11 and a
    2 cm wide handle from x = 0 to `handle_length`, with no points in
    `handle_gap` (a mesh's long triangles leave holes like that)."""
    rng = np.random.default_rng(0)
    r, a = 0.11 * np.sqrt(rng.uniform(size=800)), rng.uniform(0, 2 * np.pi, 800)
    body = np.stack([-0.11 + r * np.cos(a), r * np.sin(a), FLOOR + rng.uniform(0, 0.05, 800)], axis=1)
    x = np.linspace(0.0, handle_length, 120)
    x = x[(x < handle_gap[0]) | (x > handle_gap[1])]
    handle = np.concatenate(
        [np.stack([x, np.full_like(x, side), np.full_like(x, FLOOR + z)], axis=1)
         for side in (-0.01, 0.01) for z in (0.04, 0.055)]
    )
    points = np.concatenate([body, handle])
    c, s = np.cos(yaw), np.sin(yaw)
    points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
    return points, len(body)


@pytest.mark.parametrize("yaw", [0.0, 2.3])
def test_detects_the_whole_handle_across_holes_in_the_mesh(yaw):
    points, n_body = pan(yaw=yaw)
    mask = detect_handle(points)
    assert mask is not None
    assert mask[n_body:].all()  # every handle point, on both sides of the hole
    assert mask[:n_body].mean() < 0.02  # at most a sliver of the rim where the handle joins


def test_a_disk_has_no_handle():
    points, n_body = pan()
    assert detect_handle(points[:n_body]) is None


@pytest.mark.parametrize("yaw", [0.0, 2.3])
def test_grip_is_near_the_body_across_the_handle(yaw):
    points, n_body = pan(yaw=yaw)
    mask = np.zeros(len(points), bool)
    mask[n_body:] = True
    grasps = handle_grasps(points, mask, floor_clearance=0.023)
    assert len(grasps) == 2
    c, s = np.cos(yaw), np.sin(yaw)
    along = np.array([c, s])  # the handle's direction after the rotation in pan()
    for grasp in grasps:
        pose = grasp.tcp_pose_world
        assert pose[:3, 3][:2] @ along == pytest.approx(GRIP_FROM_BODY, abs=0.01)  # 4.5 cm out, not at the tip
        assert pose[2, 2] == pytest.approx(-1.0)  # from above
        assert abs(pose[:2, 0] @ along) < 1e-6  # fingers close across the handle
        assert grasp.width_hint == pytest.approx(0.02, abs=0.002)
        assert FLOOR + 0.04 <= pose[2, 3] <= FLOOR + 0.055


def test_a_short_handle_is_held_at_its_middle():
    points, n_body = pan(handle_length=0.06, handle_gap=(1.0, 1.0))
    mask = np.zeros(len(points), bool)
    mask[n_body:] = True
    grip = handle_grasps(points, mask, floor_clearance=0.023)[0].tcp_pose_world[:3, 3]
    assert grip[0] == pytest.approx(0.03, abs=0.01)


def test_grip_stays_clear_of_the_floor():
    points, n_body = pan()
    points[n_body:, 2] = FLOOR + 0.005  # a handle lying almost flat
    mask = np.zeros(len(points), bool)
    mask[n_body:] = True
    grip = handle_grasps(points, mask, floor_clearance=0.023)[0].tcp_pose_world[:3, 3]
    assert grip[2] == pytest.approx(FLOOR + 0.023, abs=1e-3)
