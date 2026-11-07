"""From a recorded pole-camera frame to world points (Motive frame, z up, metres): the chain in
docs/memory-map-next.md "From a pixel to the kitchen". Numpy + PyYAML only (no OpenCV), so it runs in any env.

    world <- planar(base x, y, yaw)            the base's turning centre, from the mocap body (onboard/frames.py)
          <- arm_in_base (dx, dy, dz)          the arm base in that frame: the calibration's T is relative to the arm
                                               base, the pose to the turning centre (+0.1825 m: the plate's centre in
                                               embodiments/tidybot_fr3/robot.yaml, ASSUMED; dz +0.105: real plate
                                               0.44 m vs the sim's 0.335 that T's z was solved against)
          <- T_pole                            cameras.pole.T (OpenCV camera axes: x right, y down, z forward)
          <- color camera point                depth deprojected, moved to the color camera by depth_to_color

Recordings: a pass dir from onboard/collect_stream.py (pole/{color,depth,ir1}/NNNNNN.png, pole/frames.jsonl,
pole/meta.json, state.jsonl).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from chefbot.onboard.frames import BaseFrames, wrap

# The mocap body on the base (onboard/config.yaml, HT-1a).
MARKER_YAW_OFFSET_DEG = 3.14
MARKER_XY = (0.072, -0.002)
# The arm base in the base (turning-centre) frame: plate centre x -0.1825 in the arm frame -> arm at +0.1825.
ARM_IN_BASE = (0.1825, 0.0, 0.105)
MOCAP_BODY = "chefbot-tidy"


def planar(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


def translate(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def load_pole_T(config_path) -> np.ndarray:
    """cameras.pole.T from an environments/real/config.yaml."""
    return np.asarray(yaml.safe_load(Path(config_path).read_text())["cameras"]["pole"]["T"], dtype=float)


# --- the robot's pose over time ---------------------------------------------------------------------------------------


def _vec(v):
    return [float(x) for x in (json.loads(v) if isinstance(v, str) else v)]


class Track:
    """The base pose (x, y, yaw) over host time, from state.jsonl's mocap body rows (only valid tracking)."""

    def __init__(self, t: np.ndarray, pose: np.ndarray):
        self.t, self.pose = t, pose
        self.yaw_unwrapped = np.unwrap(pose[:, 2]) if len(pose) else pose[:, 2]

    @classmethod
    def from_state(cls, path, body: str = MOCAP_BODY, frames: BaseFrames | None = None) -> "Track":
        frames = frames or BaseFrames(math.radians(MARKER_YAW_OFFSET_DEG), MARKER_XY)
        pos = ori = None
        valid = True
        ts, poses = [], []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                changed = False
                if f"{body}::pos" in d:
                    pos, changed = _vec(d[f"{body}::pos"]), True
                if f"{body}::ori" in d:
                    ori, changed = _vec(d[f"{body}::ori"]), True
                if f"{body}::tracking_valid" in d:
                    valid = str(d[f"{body}::tracking_valid"]) == "1"
                if changed and valid and pos is not None and ori is not None:
                    ts.append(float(d["wall"]))
                    poses.append(frames.base_from_mocap(pos, ori))
        return cls(np.asarray(ts), np.asarray(poses).reshape(-1, 3))

    def pose_at(self, t: float, max_gap_s: float = 0.25):
        """Interpolated (x, y, yaw) at host time t, or None if the nearest samples are more than max_gap_s away (the
        publisher only writes changes, so a still robot leaves gaps: the last pose stands if it's within the gap of a
        sample on each side, or of the last one when the robot hadn't moved)."""
        if not len(self.t):
            return None
        i = int(np.searchsorted(self.t, t))
        if i == 0:
            return self.pose[0].copy() if self.t[0] - t <= max_gap_s else None
        if i == len(self.t):
            return self.pose[-1].copy()  # nothing changed since: the robot is where it was last seen
        t0, t1 = self.t[i - 1], self.t[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        if t1 - t0 > max_gap_s:  # standing still between the two: hold the earlier sample
            a = 0.0 if t - t0 < t1 - t else 1.0
        xy = (1 - a) * self.pose[i - 1, :2] + a * self.pose[i, :2]
        yaw = wrap((1 - a) * self.yaw_unwrapped[i - 1] + a * self.yaw_unwrapped[i])
        return np.array([xy[0], xy[1], yaw])


# --- the camera ------------------------------------------------------------------------------------------------------


@dataclass
class Intrinsics:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray  # OpenCV order k1, k2, p1, p2, k3 (as onboard/calibration/camera.py takes RealSense's coeffs)

    @classmethod
    def from_meta(cls, m: dict) -> "Intrinsics":
        K = np.array([[m["fx"], 0, m["ppx"]], [0, m["fy"], m["ppy"]], [0, 0, 1.0]])
        return cls(int(m["width"]), int(m["height"]), K, np.asarray(m.get("coeffs", [0] * 5)[:5], dtype=float))

    def project(self, p: np.ndarray) -> np.ndarray:
        """Camera points (N x 3, z forward) to pixels (N x 2), OpenCV's distortion model."""
        p = np.atleast_2d(p)
        z = np.where(np.abs(p[:, 2]) < 1e-9, 1e-9, p[:, 2])
        x, y = p[:, 0] / z, p[:, 1] / z
        k1, k2, p1, p2, k3 = self.D
        if np.any(self.D):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
        return np.stack([self.K[0, 0] * x + self.K[0, 2], self.K[1, 1] * y + self.K[1, 2]], axis=1)

    def rays(self) -> np.ndarray:
        """Per-pixel (H x W x 2) normalized x/y for an undistorted camera (the D455's depth stream has zero D)."""
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]], axis=-1)


def _extrinsic(e: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(e["rotation_colmajor"], dtype=float).reshape(3, 3, order="F")
    T[:3, 3] = e["translation_m"]
    return T


@dataclass
class PoleCamera:
    color: Intrinsics
    depth: Intrinsics
    depth_scale: float
    color_from_depth: np.ndarray  # 4x4
    _rays: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_meta(cls, meta: dict) -> "PoleCamera":
        return cls(Intrinsics.from_meta(meta["color"]), Intrinsics.from_meta(meta["depth"]), float(meta["depth_scale_m"]),
                   _extrinsic(meta["depth_to_color"]))  # fmt: skip

    def depth_points(self, depth_u16: np.ndarray, stride: int = 1, max_m: float = 4.0) -> np.ndarray:
        """Valid depth pixels as points in the COLOR camera (N x 3)."""
        if self._rays is None:
            self._rays = self.depth.rays()
        d = depth_u16[::stride, ::stride].astype(np.float32) * self.depth_scale
        r = self._rays[::stride, ::stride]
        ok = (d > 0.1) & (d < max_m)
        z = d[ok]
        p = np.stack([r[..., 0][ok] * z, r[..., 1][ok] * z, z], axis=1)
        return p @ self.color_from_depth[:3, :3].T + self.color_from_depth[:3, 3]

    def aligned_xyz(self, depth_u16: np.ndarray, max_m: float = 4.0) -> np.ndarray:
        """Depth registered to the color image: H x W x 3 color-camera points (nan where no depth), the nearest point
        per color pixel (a z-buffer), so a color-image mask picks its 3D points."""
        p = self.depth_points(depth_u16, max_m=max_m)
        uv = np.round(self.color.project(p)).astype(np.int64)
        W, H = self.color.width, self.color.height
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        p, uv = p[ok], uv[ok]
        idx = uv[:, 1] * W + uv[:, 0]
        order = np.argsort(-p[:, 2], kind="stable")  # far to near: the nearest point per pixel is written last
        out = np.full((H * W, 3), np.nan, dtype=np.float32)
        out[idx[order]] = p[order]  # numpy: with repeated indices, the last assignment stands
        # depth is sparser than color (1280x720 into 1280x800 with the 0.06 m baseline): fill 1-px holes from neighbours
        out = out.reshape(H, W, 3)
        hole = np.isnan(out[..., 2])
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.roll(out, (dy, dx), axis=(0, 1))
            fill = hole & ~np.isnan(nb[..., 2])
            out[fill] = nb[fill]
            hole &= ~fill
        return out


def rot_cam(pitch: float, roll: float) -> np.ndarray:
    """A small tilt of the camera in its own axes (OpenCV: x right, y down, z forward): pitch about x, roll about z."""
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return T


def mount(T_pole: np.ndarray, x=(*ARM_IN_BASE, 0.0, 0.0, 0.0)) -> np.ndarray:
    """The pole color camera in the base (turning-centre) frame, for x = (dx, dy, dz, dyaw, dpitch, droll): the arm
    base's offset in the base frame (m) and small corrections to the calibration (deg)."""
    dx, dy, dz, dyaw, dp, dr = (float(v) for v in x)
    return translate(dx, dy, dz) @ planar(0, 0, math.radians(dyaw)) @ T_pole @ rot_cam(math.radians(dp), math.radians(dr))


MOUNT_FIT = Path(__file__).with_name("pole_mount_fit.yaml")


def load_mount(config_path, fit_path=MOUNT_FIT) -> np.ndarray:
    """cameras.pole.T from config_path, with the fitted offsets (pole_mount_fit.yaml) if there is one for that T."""
    T_pole = load_pole_T(config_path)
    if fit_path and Path(fit_path).exists():
        fit = yaml.safe_load(Path(fit_path).read_text())
        if np.allclose(np.asarray(fit["for_T"], float), T_pole, atol=1e-4):
            return mount(T_pole, fit["x"])
    return mount(T_pole)


def world_from_camera(base_pose, T_mount: np.ndarray) -> np.ndarray:
    """The pole color camera's pose in the world (4x4, OpenCV camera axes), T_mount from mount() / load_mount()."""
    return planar(*base_pose) @ T_mount


def to_world(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


# --- a recorded pass -------------------------------------------------------------------------------------------------


class Pass:
    """One recorded pass (stream/<pass>/), the pole camera's frames with the robot's pose at each."""

    def __init__(self, root, cam: str = "pole"):
        self.root = Path(root)
        self.cam_dir = self.root / cam
        self.meta = json.loads((self.cam_dir / "meta.json").read_text())
        self.camera = PoleCamera.from_meta(self.meta)
        self.frames = [json.loads(l) for l in (self.cam_dir / "frames.jsonl").read_text().splitlines() if l.strip()]
        self.track = Track.from_state(self.root / "state.jsonl")

    def name(self, frame: dict) -> str:
        return f"{int(frame['n']):06d}.png"

    def pose(self, frame: dict):
        return self.track.pose_at(float(frame["t_color_ms"]) / 1000.0)

    def color(self, frame: dict) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.cam_dir / "color" / self.name(frame)).convert("RGB"))

    def depth(self, frame: dict) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.cam_dir / "depth" / self.name(frame)))

    def ir(self, frame: dict, which: int = 1) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.cam_dir / f"ir{which}" / self.name(frame)))


# --- the kitchen's counters (maps/<map>/spec.yaml boxes) -------------------------------------------------------------


@dataclass
class Counter:
    name: str
    fixture: str
    center: np.ndarray
    size: np.ndarray  # length (box x), depth (box y)
    yaw: float
    height: float

    def local(self, xy: np.ndarray) -> np.ndarray:
        """World x/y (N x 2) into the box's frame (centred, axes along its sides)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        d = np.atleast_2d(xy) - self.center
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
        q = self.local(xy)
        return (np.abs(q[:, 0]) <= self.size[0] / 2 + margin) & (np.abs(q[:, 1]) <= self.size[1] / 2 + margin)


def load_counters(spec_path, fixtures_only: bool = True) -> list[Counter]:
    """The spec's box obstacles with a height (counters, fridge...); with fixtures_only, those that are fixtures (not
    no-go floor zones)."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    out = []
    for o in spec.get("obstacles", []):
        if "box" not in o or "height_m" not in o or (fixtures_only and "fixture" not in o):
            continue
        b = o["box"]
        out.append(Counter(o["name"], o.get("fixture", ""), np.asarray(b["center"], float), np.asarray(b["size"], float),
                           math.radians(float(b.get("yaw_deg", 0.0))), float(o["height_m"])))  # fmt: skip
    return out
