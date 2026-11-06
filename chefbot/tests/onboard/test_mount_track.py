"""The pole camera's mount tracker on synthetic frames of the hand's printed card."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from chefbot.interface.mount_track import MountTracker
from chefbot.onboard.calibration import fr3, targets
from chefbot.onboard.calibration.wrist import _T, _rot, ik

MOUNT = _T(np.eye(3), [0, 0, 0.335])
K = np.array([[383.0, 0, 320], [0, 383.0, 240], [0, 0, 1]])
Q0 = np.array([0.0, -0.73, 0.0, -2.18, 0.0, 1.45, 0.75])
# the pole camera as calibrated on 2026-11-03 (base frame)
Y = np.array([[-0.22954, -0.17202, 0.95798, -0.24949], [-0.95765, -0.13588, -0.25386, 0.2612],
              [0.17383, -0.97568, -0.13355, 1.0381], [0, 0, 0, 1]])  # fmt: skip


def flange_in_base(q):
    return MOUNT @ fr3.flange(q)


def apart(A, B):
    return np.linalg.norm(A[:3, 3] - B[:3, 3]), np.degrees(np.linalg.norm(cv2.Rodrigues(A[:3, :3] @ B[:3, :3].T)[0]))


@pytest.fixture(scope="module")
def scene():
    F0 = flange_in_base(Q0)  # the card 8 cm out along the flange's z, facing the camera at Q0
    p = F0[:3, 3] + F0[:3, 2] * 0.08
    z = (Y[:3, 3] - p) / np.linalg.norm(Y[:3, 3] - p)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    Z = np.linalg.inv(F0) @ _T(np.stack([x, np.cross(z, x), z], 1), p)
    page, points = targets.hand_markers(), targets.hand_card_points()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    c, ids, _ = det.detectMarkers(page)
    H, _ = cv2.findHomography(np.concatenate([points[i][:, :2] for i in ids.ravel()]), np.concatenate([q[0] for q in c]))
    rng = np.random.default_rng(0)

    def render(q, Ytrue):
        C = np.linalg.inv(Ytrue) @ flange_in_base(q) @ Z
        patch = np.array([[-0.08, 0.045], [0.08, 0.045], [0.08, -0.045], [-0.08, -0.045]])
        cam = [(K @ (C[:3, :3] @ [u, v, 0] + C[:3, 3]))[:2] / (C[:3, :3] @ [u, v, 0] + C[:3, 3])[2] for u, v in patch]
        M = cv2.getPerspectiveTransform(cv2.perspectiveTransform(patch[None], H)[0].astype(np.float32), np.array(cam, np.float32))
        img = cv2.warpPerspective(page, M, (640, 480), flags=cv2.INTER_AREA, borderValue=128)
        return cv2.GaussianBlur(np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8), (3, 3), 0.6)

    def hand_at(d):
        T = fr3.flange(Q0)
        T[:3, 3] += d
        return ik(Q0, T)

    spread = [q for q in (hand_at(rng.uniform(-0.15, 0.15, 3) * [1, 1, 0.6]) for _ in range(20)) if q is not None]
    return Z, points, render, spread


def test_fits_the_camera_from_the_hand_spread(scene):
    Z, points, render, spread = scene
    start = Y @ _T(_rot([0, 1, 0], 1.0), [0.015, 0, 0])  # the config 1.5 cm, 1 deg off
    t = MountTracker(Z, points, pose=start)
    for q in spread:
        t.update(render(q, Y), K, q, np.zeros(7), flange_in_base)
    m, deg = apart(t.pose, Y)
    assert t.full and m < 0.003 and deg < 0.2


def test_ignores_frames_while_the_arm_moves(scene):
    Z, points, render, spread = scene
    t = MountTracker(Z, points, pose=Y)
    assert not t.update(render(spread[0], Y), K, spread[0], np.full(7, 0.1), flange_in_base)


def test_takes_up_a_knocked_camera(scene):
    Z, points, render, spread = scene
    t = MountTracker(Z, points, pose=Y)
    for q in spread[:10]:
        t.update(render(q, Y), K, q, np.zeros(7), flange_in_base)
    knocked = Y @ _T(_rot([1, 0, 0], 2.0), [0.02, 0, 0])
    for q in spread[:12]:
        t.update(render(q, knocked), K, q, np.zeros(7), flange_in_base)
    m, deg = apart(t.pose, knocked)
    assert t.moved and m < 0.003 and deg < 0.2
