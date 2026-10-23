"""The real TidyBot++ as a fridge-lab host: what episode.run_skill_episode drives in place of the simulator.

Same contract as kitchen.LabHost (skill_api/fridge_robot.py): odom is the robot root (the arm base frame)
where the episode started; each control step takes an absolute 11-D target [flange pose in odom,
gripper, root x, y, yaw in odom]. The difference is time: it is real. A control thread runs at 15 Hz
from the start of the episode, each tick taking one step toward the latest target, so the robot keeps
going (holding its last command) while the skill thinks or waits on a model, as the simulator does.

Each tick, as the simulator's controller (experiments/astranomy/fridge.py) does per step, scaled by
safety.speed: the base is sent the root target (kept within the geofence); the arm takes one clipped
Cartesian step toward the flange target relative to where the base is now (so a held handle stays
held while the base moves), solved by IK and clipped per joint, never below safety.flange_z_min.
"""

from __future__ import annotations

import math
import pickle
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.fridge_lab.real.arms import make_arm
from experiments.fridge_lab.real.base import make_base
from experiments.fridge_lab.real.cameras import H, W, make_camera

HZ = 15.0
# The simulator's limits per 20 Hz IK tick (experiments/astranomy/host.py), as limits per 15 Hz step.
POS_CLIP, ROT_CLIP, JOINT_CLIP = 0.02 * 20 / 15, 0.1 * 20 / 15, 0.05 * 20 / 15
TRACKING_SLACK = 0.05  # rad: step from the last command while the arm follows it, else from where it is
# (slack + one full-speed joint step stays under the arm server's 0.15 rad refusal limit)
ZED_CALIB = {"pole": "robot0_pole_zed", "base": "robot0_base_zed"}  # MessyMem's calibration, by the sim's names

Pose = namedtuple("Pose", "x y yaw")


def se2(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:3, 3] = [x, y, z]
    return T


def xy_yaw(T: np.ndarray) -> tuple[float, float, float]:
    return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pose_of_row(row) -> np.ndarray:
    T = np.eye(4)
    w, x, y, z = row[3:7]
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = row[:3]
    return T


def quat_wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


class RobotFault(RuntimeError):
    pass


class RealHost:
    realtime = True

    def __init__(self, cfg: dict, out_dir: Path, record: bool = True, log=print):
        self.cfg, self.out, self.record, self.log = cfg, Path(out_dir), record, log
        s = cfg["safety"]
        self.speed, self.z_min = float(s["speed"]), float(s["flange_z_min"])
        self.geofence, self.yaw_limit = float(cfg["base"]["geofence_m"]), math.radians(float(cfg["base"]["yaw_limit_deg"]))
        self.root_in_base = se2(0, 0, 0)
        self.root_in_base[:3, 3] = cfg["arm"]["root_in_base"]
        self.arm, self.base = make_arm(cfg), make_base(cfg)
        self.kin = self.arm.kin
        self.cams = {n: make_camera(n, c, bool(cfg.get("dry_run"))) for n, c in cfg["cameras"].items()}
        self.mounts = {n: self._mount(n, c) for n, c in cfg["cameras"].items()}
        self.rpc = threading.Lock()  # the RPC proxies are not thread-safe
        self.cond = threading.Condition()
        self.step_count, self.frames, self.success_deg = 0, [], 60.0
        self._thread, self._stop, self.fault = None, False, None
        self._clock_on = False  # robot time counts from the skill's first request (process start-up is free, as in the sim)
        self._row, self._rec, self._q_cmd, self._meas = None, None, None, None

    # -- calibration -----------------------------------------------------------------

    def _mount(self, name: str, c: dict) -> tuple[str, np.ndarray, bool]:
        """(parent "root" or "flange", camera pose in it with OpenCV axes, calibrated?)"""
        if c["type"] == "usb":
            if c.get("cam_in_flange") is not None:
                return "flange", np.asarray(c["cam_in_flange"], dtype=float), True
            # Uncalibrated: looking along the fingers from 6 cm above the flange (image right = flange -y).
            T = np.eye(4)
            T[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
            T[:3, 3] = [0.0, 0.0, 0.06]
            return "flange", T, False
        e = c.get("extrinsics")
        if e is None:
            from chefbot.envs.robocasa.yam_tidybot import ZED_IN_ARM

            T = ZED_IN_ARM[ZED_CALIB[name]]
        elif isinstance(e, str):
            with open(e, "rb") as f:
                T = pickle.load(f)
        else:
            T = e
        T = np.asarray(T, dtype=float)
        T[:3, :3] = Rotation.from_matrix(T[:3, :3]).as_matrix()  # re-orthonormalize
        return "root", T, True

    def uncalibrated(self) -> list[str]:
        out = [f"{n} pose" for n, (_, _, ok) in self.mounts.items() if not ok]
        return out + [f"{n} intrinsics" for n, c in self.cams.items() if not getattr(c, "calibrated", True)]

    # -- frames ----------------------------------------------------------------------

    def _root_in_vehicle(self, b) -> np.ndarray:
        return se2(*b) @ self.root_in_base

    def _measure(self) -> dict:
        with self.rpc:
            q, width = self.arm.state()
            b = self.base.pose()
        root = self._root_in_vehicle(b)
        return {"q": q, "width": width, "b": b, "root": root, "root_odom": np.linalg.inv(self.odom) @ root, "t": time.time()}

    # -- episode ---------------------------------------------------------------------

    def reset_robot(self) -> None:
        """Before an episode: base controller (re)started with odometry at zero, arm to its ready pose."""
        with self.rpc:
            self.base.reset()
            self.arm.reset(self.cfg["arm"].get("ready_q"))

    def find_start(self, door=None, seed=None, **_) -> Pose:
        """The robot starts wherever the operator put it: odom is fixed here, and control begins."""
        b = self.base.pose()
        self.odom = self._root_in_vehicle(b)
        self.step_count, self.frames, self.fault, self._stop = 0, [], None, False
        self._row, self._rec, self._q_cmd, self._clock_on = None, None, None, False
        self._meas = self._measure()
        self._thread = threading.Thread(target=self._control, daemon=True)
        self._thread.start()
        if self.record:
            threading.Thread(target=self._recorder, daemon=True).start()
        return Pose(*map(float, b))

    def _hold_row(self) -> list[float]:
        m = self._meas
        F = m["root_odom"] @ self.kin.fk(m["q"])
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), 0.0 if m["width"] > 0.5 * self.kin.open_width else 1.0, *xy_yaw(m["root_odom"])]

    def _control(self) -> None:
        t_next = time.time()
        hold = self._hold_row()
        while not self._stop:
            t_next += 1.0 / HZ
            try:
                rec = self._tick(self._row or hold)
            except Exception as e:  # noqa: BLE001
                self.fault = e
                self.log(f"[real] control fault: {type(e).__name__}: {e}; holding")
                self.halt()
                with self.cond:
                    self.cond.notify_all()
                return
            with self.cond:
                self.step_count += self._clock_on
                self._rec = rec
                self.cond.notify_all()
            dt = t_next - time.time()
            if dt > 0:
                time.sleep(dt)
            elif dt < -0.5:  # fell far behind: don't try to catch up
                t_next = time.time()

    def _tick(self, row) -> dict:
        row = np.asarray(row, dtype=float)
        m = self._meas = self._measure()
        # Base: the root target in odom, kept inside the geofence, as a base-center target.
        tx, ty, tyaw = row[8:11]
        d = math.hypot(tx, ty)
        if d > self.geofence:
            tx, ty = tx * self.geofence / d, ty * self.geofence / d
        tyaw = float(np.clip(wrap(tyaw), -self.yaw_limit, self.yaw_limit))
        center = self.odom @ se2(tx, ty, tyaw) @ np.linalg.inv(self.root_in_base)
        # Arm: one clipped step toward the flange target, relative to where the base is now.
        F_target = np.linalg.inv(m["root"]) @ self.odom @ pose_of_row(row)
        q_meas = m["q"]
        q = self._q_cmd if self._q_cmd is not None and np.abs(self._q_cmd - q_meas).max() < TRACKING_SLACK else q_meas
        F = self.kin.fk(q)
        dp = F_target[:3, 3] - F[:3, 3]
        if np.linalg.norm(dp) > POS_CLIP * self.speed:
            dp *= POS_CLIP * self.speed / np.linalg.norm(dp)
        rot = Rotation.from_matrix(F_target[:3, :3] @ F[:3, :3].T).as_rotvec()
        if np.linalg.norm(rot) > ROT_CLIP * self.speed:
            rot *= ROT_CLIP * self.speed / np.linalg.norm(rot)
        Fd = np.eye(4)
        Fd[:3, :3] = Rotation.from_rotvec(rot).as_matrix() @ F[:3, :3]
        Fd[:3, 3] = F[:3, 3] + dp
        Fd[2, 3] = max(Fd[2, 3], self.z_min)
        q_ik = self.kin.ik(Fd, q)
        ik_ok = q_ik is not None
        if ik_ok:
            q = q + np.clip(q_ik - q, -JOINT_CLIP * self.speed, JOINT_CLIP * self.speed)
        width = 0.0 if row[7] >= 0.5 else self.kin.open_width
        with self.rpc:
            self.base.drive_to(xy_yaw(center))
            sent = self.arm.command(q, width)
        if not sent:
            raise RobotFault("the arm server refused a joint target far from the measured joints")
        self._q_cmd = q
        Fm = m["root_odom"] @ self.kin.fk(q_meas)
        bx, by, byaw = xy_yaw(m["root_odom"])
        return {"pos_error_m": float(np.linalg.norm(row[:3] - Fm[:3, 3])),
                "rot_error_rad": float(np.linalg.norm(Rotation.from_matrix(pose_of_row(row)[:3, :3] @ Fm[:3, :3].T).as_rotvec())),
                "base_error_m": float(math.hypot(row[8] - bx, row[9] - by)), "ik_ok": bool(ik_ok)}  # fmt: skip

    def step(self, row) -> dict:
        """Make `row` the target and return after the next control tick."""
        if self.fault is not None:
            raise RobotFault(f"robot halted: {self.fault}")
        self._clock_on = True
        with self.cond:
            self._row = [float(v) for v in row]
            n = self.step_count
            if not self.cond.wait_for(lambda: self.step_count > n or self.fault is not None, timeout=2.0):
                raise RobotFault("control loop stalled")
        if self.fault is not None:
            raise RobotFault(f"robot halted: {self.fault}")
        return self._rec

    def halt(self) -> None:
        """Stop: the base holds where it is, the arm holds its joints."""
        self._stop = True
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)  # no tick in flight after the hold
        try:
            with self.rpc:
                b = self.base.pose()
                self.base.drive_to(b)
                self.arm.hold()
        except Exception as e:  # noqa: BLE001
            self.log(f"[real] halt: {type(e).__name__}: {e}")

    def close(self) -> None:
        self.halt()
        for c in self.cams.values():
            c.close()
        with self.rpc:
            self.arm.close()
            self.base.close()

    # -- what the skill sees -------------------------------------------------------------

    def proprio(self) -> dict:
        self._clock_on = True
        m = self._meas if self._thread is not None and self._thread.is_alive() else self._measure()
        F = m["root_odom"] @ self.kin.fk(m["q"])
        closure = float(np.clip(1.0 - m["width"] / self.kin.open_width, 0.0, 1.0))
        return {"step": self.step_count, "sim_time_s": self.step_count / HZ, "flange_pose_odom": F.tolist(),
                "base_pose_odom": list(xy_yaw(m["root_odom"])), "gripper_closed_cmd": bool(self._row and self._row[7] >= 0.5),
                "gripper_closure": closure, "gripper_width_m": float(m["width"]), "arm_q": [float(v) for v in m["q"]],
                "arm_q_cmd": None if self._q_cmd is None else [float(v) for v in self._q_cmd],
                "base_in_contact": False}  # no contact sensing on the real base  # fmt: skip

    def rgbd(self) -> dict[str, dict]:
        self._clock_on = True
        m = self._measure()
        inv = np.linalg.inv(self.odom)
        out = {}
        for name, cam in self.cams.items():
            rgb, depth, _ = cam.frame()
            parent, T, _ = self.mounts[name]
            to_odom = inv @ m["root"] @ (self.kin.fk(m["q"]) if parent == "flange" else np.eye(4)) @ T
            out[name] = {"rgb": rgb, "depth": depth, "K": np.asarray(cam.K), "D": cam.D, "model": cam.model, "cam_to_odom": to_odom}
        return out

    def debug(self) -> dict:
        return {"doors": [], "robot_touching": []}  # no ground truth on a real robot

    def door_deg(self) -> None:
        return None  # nothing measures the door on the real robot

    # -- recording -------------------------------------------------------------------

    def _recorder(self) -> None:
        """10 fps of the cameras, 2x2, while the episode runs."""
        while not self._stop and self.fault is None:
            t = time.time()
            views = []
            for cam in list(self.cams.values())[:4]:
                try:
                    views.append(np.ascontiguousarray(cam.frame()[0][::2, ::2]))
                except Exception:  # noqa: BLE001
                    views.append(np.zeros((H // 2, W // 2, 3), np.uint8))
            while len(views) < 4:
                views.append(np.zeros((H // 2, W // 2, 3), np.uint8))
            self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:], 1)], 0))
            time.sleep(max(0.0, 0.1 - (time.time() - t)))

    def save_video(self, name: str = "video.mp4") -> Path | None:
        import imageio.v3 as iio

        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=10, codec="libx264")
        return p
