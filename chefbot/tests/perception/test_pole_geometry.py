import json
import math

import numpy as np

from chefbot.perception.pole_geometry import (ARM_IN_BASE, Counter, Intrinsics, PoleCamera, Track, mount, planar,
                                              to_world, world_from_camera)  # fmt: skip


def _cam(D=(0, 0, 0, 0, 0)):
    return Intrinsics(64, 48, np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]]), np.asarray(D, float))


def test_project_pinhole_centre_and_offset():
    c = _cam()
    uv = c.project(np.array([[0, 0, 2.0], [0.2, -0.1, 1.0]]))
    np.testing.assert_allclose(uv, [[32, 24], [42, 19]])


def test_depth_points_identity_extrinsic_round_trip():
    depth = Intrinsics(64, 48, np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]]), np.zeros(5))
    cam = PoleCamera(_cam(), depth, 0.001, np.eye(4))
    d = np.full((48, 64), 1500, np.uint16)
    p = cam.depth_points(d)
    assert p.shape == (48 * 64, 3) and np.allclose(p[:, 2], 1.5)
    uv = cam.color.project(p)
    u, v = np.meshgrid(np.arange(64), np.arange(48))
    np.testing.assert_allclose(uv, np.stack([u.ravel(), v.ravel()], 1), atol=1e-6)
    xyz = cam.aligned_xyz(d)
    assert np.allclose(xyz[..., 2], 1.5)


def test_track_interpolates_and_unwraps_yaw(tmp_path):
    rows = [{"wall": 0.0, "chefbot-tidy::tracking_valid": "1"}]
    for t, yaw in ((0.0, math.pi - 0.05), (0.1, -math.pi + 0.05)):  # crossing +-pi
        q = [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]
        rows.append({"wall": t, "chefbot-tidy::pos": json.dumps([1.0, 2.0, 0.4]), "chefbot-tidy::ori": json.dumps(q)})
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    from chefbot.onboard.frames import BaseFrames

    tr = Track.from_state(tmp_path / "s.jsonl", frames=BaseFrames(0.0, (0.0, 0.0)))
    p = tr.pose_at(0.05)
    assert abs(abs(p[2]) - math.pi) < 1e-6  # half way across the wrap, not through 0
    np.testing.assert_allclose(p[:2], [1.0, 2.0])
    assert tr.pose_at(-1.0) is None


def test_mount_default_is_arm_offset_then_T():
    T = planar(0.1, 0.2, 0.3, 1.0)
    M = mount(T)
    np.testing.assert_allclose(M[:3, 3], np.array([0.1, 0.2, 1.0]) + np.array(ARM_IN_BASE))
    W = world_from_camera((1.0, 0.0, math.pi / 2), M)
    p = to_world(W, np.zeros((1, 3)))[0]
    np.testing.assert_allclose(p, [1.0 - (0.2 + ARM_IN_BASE[1]), 0.1 + ARM_IN_BASE[0], 1.0 + ARM_IN_BASE[2]], atol=1e-9)


def test_counter_contains_rotated():
    c = Counter("c", "counter", np.array([0.0, 0.0]), np.array([2.0, 0.5]), math.radians(90), 0.9)
    assert c.contains(np.array([[0.0, 0.9]]))[0] and not c.contains(np.array([[0.9, 0.0]]))[0]


def test_aligned_xyz_keeps_the_nearest_point_per_pixel():
    depth = Intrinsics(64, 48, np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]]), np.zeros(5))
    # two depth pixels landing on the same color pixel: a color camera with a quarter of the focal length
    color = Intrinsics(16, 12, np.array([[12.5, 0, 8], [0, 12.5, 6], [0, 0, 1]]), np.zeros(5))
    cam = PoleCamera(color, depth, 0.001, np.eye(4))
    d = np.full((48, 64), 2000, np.uint16)
    d[::2, ::2] = 1000  # every 4th pixel nearer
    xyz = cam.aligned_xyz(d)
    assert np.nanmax(xyz[..., 2]) == 1.0
