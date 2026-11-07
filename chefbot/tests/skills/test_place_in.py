"""place_in/v1 on a fake skill robot (its chefbot-v1 calls, no sandbox): a counter at 0.90 m with a bowl's rim on it
(0.07 m radius, 0.98 m up), a wrist camera looking along the hand, the arm going where it is sent. It finds the rim
from far, looks again from close above, releases over the rim's centre just above it, and lifts the hand away."""

import importlib.util
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from chefbot import skills

COUNTER_Z, RIM_Z, RIM_R, BOWL = 0.90, 0.98, 0.07, np.array([0.62, 0.05])
CAM_IN_FLANGE = np.eye(4)
CAM_IN_FLANGE[:3, 3] = [0.05, 0.0, 0.0]  # beside the hand, looking along it (the flange's z)
K = np.array([[380.0, 0, 320], [0, 380.0, 240], [0, 0, 1]])


class FakeSkillRobot:
    def __init__(self, held=True):
        self.F = np.eye(4)
        self.F[:3, :3] = Rotation.from_euler("y", 150, degrees=True).as_matrix()  # forward and down, as from READY
        self.F[:3, 3] = [0.10, 0.0, 1.30]
        self.width, self.opened, self.log_lines = (0.03 if held else 0.0), [], []

    def proprio(self):
        return {"flange_pose_odom": self.F.tolist(), "flange_pose_root": self.F.tolist(), "base_pose_odom": [0.0, 0.0, 0.0],
                "gripper_width_m": self.width, "gripper_closed_cmd": True, "arm_q": [0.0] * 7, "arm_q_cmd": [0.0] * 7}  # fmt: skip

    def _depth(self, T):
        v, u = np.indices((480, 640))
        rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) @ T[:3, :3].T
        o = T[:3, 3]
        depth = np.zeros((480, 640))
        for z in (COUNTER_Z, RIM_Z):  # the rim ring over the counter: the nearer one where they overlap
            t = (z - o[2]) / rays[..., 2]
            xy = o[:2] + t[..., None] * rays[..., :2]
            r = np.linalg.norm(xy - BOWL, axis=-1)
            hit = (t > 0) if z == COUNTER_Z else (t > 0) & (np.abs(r - RIM_R) < 0.006)
            depth = np.where(hit, t, depth)  # t is depth along the optical axis (the ray's z in the camera is 1)
        return depth

    def observe(self, cameras=None):
        T = self.F @ CAM_IN_FLANGE
        cams = {"wrist": {"rgb": np.zeros((480, 640, 3), np.uint8), "depth": self._depth(T), "K": K, "cam_to_odom": T}}
        return {"proprio": self.proprio(), "cameras": cams}

    def _bowl_pixels(self, T):
        v, u = np.indices((480, 640))
        d = self._depth(T)
        P = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1) @ T[:3, :3].T + T[:3, 3]
        return (np.linalg.norm(P[..., :2] - BOWL, axis=-1) < RIM_R + 0.01) & (d > 0)

    def detect(self, image, queries, model="owlv2", threshold=0.1):
        m = self._bowl_pixels(self.F @ CAM_IN_FLANGE)
        if not m.any():
            return []
        v, u = np.nonzero(m)
        return [{"label": queries[0], "score": 0.7, "box": [float(u.min()), float(v.min()), float(u.max()), float(v.max())]}]

    def segment(self, image, boxes=None, points=None, point_labels=None):
        return [self._bowl_pixels(self.F @ CAM_IN_FLANGE)]

    def act_arm(self, rows):
        r = rows[-1]
        self.F = np.eye(4)
        self.F[:3, :3] = Rotation.from_quat([r[4], r[5], r[6], r[3]]).as_matrix()
        self.F[:3, 3] = r[:3]
        return {"steps": [{"pos_error_m": 0.0, "rot_error_rad": 0.0}], "proprio": self.proprio()}

    def act_joints(self, rows):
        return {"steps": [{"pos_error_m": 0.0, "rot_error_rad": 0.0}], "proprio": self.proprio()}

    def gripper_move(self, width_m, speed_mps=0.05):
        self.width = width_m
        self.opened.append(self.F[:3, 3].copy())
        return {"proprio": self.proprio()}

    def log(self, *msg):
        self.log_lines.append(" ".join(map(str, msg)))

    def save_image(self, name, image):
        pass


def _skill():
    d = skills.load("place_in/v1").dir
    sys.path.insert(0, str(d))
    try:
        spec = importlib.util.spec_from_file_location("place_in_main", d / "main.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(d))


def test_it_releases_over_the_rims_centre_just_above_it_and_lifts_away():
    m, robot = _skill(), FakeSkillRobot()
    r = m.run(robot, target="bowl", object="tomato")
    assert r["ok"], (r, robot.log_lines)
    assert r["rim"]["camera"] == "wrist" and r["rim"]["distance"] <= m.CLOSE_VIEW_M  # measured from the close view
    assert np.allclose(r["rim"]["center"][:2], BOWL, atol=0.01) and abs(r["rim"]["top"] - RIM_Z) < 0.01
    (at,) = robot.opened
    pads = at + m.PADS * robot.F[:3, 2]  # along the hand, pointing down as it opens
    assert np.allclose(at[:2], BOWL, atol=0.01) and RIM_Z + 0.04 < pads[2] < RIM_Z + 0.08
    assert robot.F[2, 3] > at[2] + 0.12  # lifted away after


def test_it_refuses_with_nothing_held():
    m = _skill()
    assert not m.run(FakeSkillRobot(held=False))["ok"]
