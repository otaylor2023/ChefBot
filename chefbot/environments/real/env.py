"""RealEnv: the real robot as an Env, through the onboard server (chefbot/onboard) on the robot.

The robot's own drivers do the control: the base drives to pose targets (set_base_target), the
arm's controller moves to poses (set_arm_pose_target) or joint targets. State arrives at a fixed
rate and get_state returns the latest; commands go out without waiting. While nothing else is
sent, a heartbeat keeps the server's watchdog from holding the robot.

Frames: arm targets go out in the frame the arm's controller uses (its point, in the arm base
frame: `flange_to_control`), so the server passes them through. Base poses are in the world
(mocap) frame; the server ties that to the base's odometry. Camera poses come from their mounts
here (`cameras`), as the server sends images only.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from chefbot.interface.env import Env
from chefbot.interface.transport import onboard as P
from chefbot.interface.transport.protocol import pack, unpack
from chefbot.interface.types import RGBD, EnvState, Pose2D

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def planar(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:3, 3] = [x, y, z]
    return T


class RobotHalted(RuntimeError):
    """The onboard server held the robot (watchdog, or a halt)."""


class RealEnv(Env):
    def __init__(
        self,
        url: str,
        flange_to_control: np.ndarray,
        cameras: dict | None = None,
        flange_in_base=None,
        velocity_horizon_s: float = 0.5,
        heartbeat_s: float = 0.1,
        max_state_age_s: float = 0.5,
        open_timeout_s: float = 10.0,
        trackers: dict | None = None,
    ):
        """`cameras`: name -> {parent: "base" | "flange", T: 4x4 camera (OpenCV axes) in the parent}.
        `flange_in_base(q) -> 4x4` places the flange for wrist cameras. `trackers`: name -> MountTracker (mount_track.py)
        for base cameras whose mount it keeps current from the hand's marker card in their frames."""
        self.flange_to_control = np.asarray(flange_to_control, dtype=float)
        self.camera_mounts = cameras or {}
        self.trackers = trackers or {}
        self._flange_in_base = flange_in_base
        self.velocity_horizon_s = velocity_horizon_s
        self.max_state_age_s = max_state_age_s
        # Entered here and closed in close(): the connection outlives this call.
        self._conn = connect(url, max_size=None, open_timeout=open_timeout_s).__enter__()
        hello = unpack(self._conn.recv(timeout=open_timeout_s))
        if hello.get("type") != P.HELLO:
            self._conn.close()
            raise RuntimeError(f"onboard server refused: {hello.get('message', hello)}")
        if hello.get("version") != P.VERSION:
            self._conn.close()
            raise RuntimeError(f"onboard server speaks protocol {hello.get('version')}, this client {P.VERSION}")
        self.server_cameras: list[str] = hello.get("cameras", [])
        self.server_limits: dict = hello.get("limits", {})
        self._t0 = time.monotonic()
        self._send_lock = threading.Lock()
        self._last_send = time.monotonic()
        self._state: dict | None = None
        self._state_at = 0.0
        self._first_state = threading.Event()
        self._replies: dict[int, dict] = {}
        self._reply_ready = threading.Condition()
        self._ids = itertools.count()
        self._closed = threading.Event()
        self.halted: str | None = None  # why the server last held the robot
        self.errors: list[str] = []
        self._gripper_closed: bool | None = None
        self._base_holding = False
        self._next_tick: float | None = None
        self._reader = threading.Thread(target=self._read, name="real-env-reader", daemon=True)
        self._reader.start()
        self._heartbeat = threading.Thread(target=self._beat, args=(heartbeat_s,), name="real-env-heartbeat", daemon=True)
        self._heartbeat.start()

    @classmethod
    def from_config(cls, flange_to_control: np.ndarray, path: str | Path = CONFIG, url: str | None = None, flange_in_base=None) -> "RealEnv":
        c = load_config(path)
        cameras = {name: {"parent": m["parent"], "T": np.asarray(m["T"], dtype=float)} for name, m in (c.get("cameras") or {}).items()}
        trackers = {}
        for name, m in (c.get("cameras") or {}).items():
            if m.get("track") and m["parent"] == "base" and flange_in_base is not None:
                from chefbot.interface.markers import hand_card_points
                from chefbot.interface.mount_track import MountTracker

                trackers[name] = MountTracker(m["track"]["T_flange_card"], hand_card_points(), pose=cameras[name]["T"])
        return cls(url or c["url"], flange_to_control, cameras, flange_in_base, c["velocity_horizon_s"], c["heartbeat_s"],
                   c["max_state_age_s"], trackers=trackers)  # fmt: skip

    # -- state ------------------------------------------------------------------

    def raw_state(self, timeout_s: float = 5.0) -> dict:
        """The latest state message as the server sent it."""
        if not self._first_state.wait(timeout_s):
            raise TimeoutError("no state from the onboard server")
        return self._state

    def get_state(self) -> EnvState:
        s = self.raw_state()
        age = time.monotonic() - self._state_at
        if age > self.max_state_age_s:
            raise RuntimeError(f"the robot's state is {age:.2f} s old: lost the onboard server?")
        if s["base_pose"] is None or s["arm_q"] is None:
            raise RuntimeError("the robot reports no base pose or no joints: are the base and arm drivers up?")
        n = len(s["arm_q"])
        return EnvState(
            base_pose=Pose2D(*(float(v) for v in s["base_pose"])),
            base_velocity=np.zeros(3) if s["base_velocity"] is None else np.asarray(s["base_velocity"], dtype=float),
            base_in_contact=False,  # not sensed
            time=time.monotonic() - self._t0,
            arm_q=np.asarray(s["arm_q"], dtype=float),
            arm_qd=np.zeros(n) if s["arm_qd"] is None else np.asarray(s["arm_qd"], dtype=float),
            gripper_width=float(s["gripper_width"] or 0.0),
            gripper_closed_cmd=bool(self._gripper_closed),
            arm_in_contact=False,  # not sensed yet (the Franka's external torque estimate, later)
        )

    # -- commands ---------------------------------------------------------------

    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        """The base drives to poses; a velocity becomes a target `velocity_horizon_s` ahead, and zero a hold."""
        if abs(vx) < 1e-6 and abs(vy) < 1e-6 and abs(vyaw) < 1e-6:
            if not self._base_holding:
                self._send({"op": P.BASE_HOLD})
                self._base_holding = True
            return
        pose = self.get_state().base_pose
        if frame == "base":
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            vx, vy = c * vx - s * vy, s * vx + c * vy
        elif frame != "world":
            raise ValueError(f"unknown frame {frame!r}")
        h = self.velocity_horizon_s
        target = Pose2D(pose.x + vx * h, pose.y + vy * h, pose.yaw + vyaw * h)
        self.set_base_target(target, (math.hypot(vx, vy), math.hypot(vx, vy), abs(vyaw)))

    def set_base_target(self, pose: Pose2D, max_vel) -> bool:
        self._send({"op": P.BASE_TARGET, "pose": [pose.x, pose.y, pose.yaw], "max_vel": [float(v) for v in max_vel]})
        self._base_holding = False
        return True

    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        self._send({"op": P.ARM_JOINTS, "q": np.asarray(q, dtype=float)})

    def set_arm_pose_target(self, flange_pose: np.ndarray) -> bool:
        control = np.asarray(flange_pose, dtype=float) @ self.flange_to_control
        self._send({"op": P.ARM_POSE, "position": control[:3, 3].copy(), "orientation": control[:3, :3].copy()})
        return True

    def set_gripper(self, closed: float) -> None:
        closed = float(closed) >= 0.5
        if closed != self._gripper_closed:
            self._send({"op": P.GRIPPER, "closed": closed})
            self._gripper_closed = closed

    def gripper_move(self, width: float, speed: float) -> None:
        self._send({"op": P.GRIPPER_MOVE, "width": float(width), "speed": float(speed)})
        self._gripper_closed = False

    def gripper_grasp(self, speed: float, force: float) -> None:
        self._send({"op": P.GRIPPER_GRASP, "speed": float(speed), "force": float(force)})
        self._gripper_closed = True

    def halt(self, reason: str = "asked") -> None:
        self._send({"op": P.HALT, "reason": reason})

    # -- cameras ----------------------------------------------------------------

    def get_camera_rgbd(self, cam: str, timeout_s: float = 10.0) -> RGBD:
        if cam not in self.camera_mounts:
            raise KeyError(f"no mount for camera {cam!r} in envs/real config (have: {sorted(self.camera_mounts)})")
        request_id = next(self._ids)
        self._send({"op": P.RGBD, "id": request_id, "camera": cam})
        with self._reply_ready:
            if not self._reply_ready.wait_for(lambda: request_id in self._replies, timeout_s):
                raise TimeoutError(f"no frame from camera {cam!r} in {timeout_s} s")
            reply = self._replies.pop(request_id)
        if not reply["ok"]:
            raise RuntimeError(f"camera {cam!r}: {reply['error']}")
        if cam in self.trackers:
            self._track(cam, reply)
        return RGBD(reply["rgb"], reply["depth"], reply["intrinsics"], self.camera_pose(cam))

    def _track(self, cam: str, reply: dict) -> None:
        """The camera's mount from its tracker (the hand's card in this frame), when it has one; never fails the frame."""
        tracker = self.trackers[cam]
        try:
            state = self.get_state()
            tracker.update(reply["rgb"], reply["intrinsics"], state.arm_q, state.arm_qd, self._flange_in_base)
            if tracker.pose is not None:
                self.camera_mounts[cam]["T"] = tracker.pose
        except Exception as exc:  # the config's (or the last tracked) mount stands
            log.warning("tracking camera %r: %s: %s", cam, type(exc).__name__, exc)

    def camera_pose(self, cam: str) -> np.ndarray:
        """Camera to world (OpenCV axes), from its mount and where the robot is now."""
        mount = self.camera_mounts[cam]
        state = self.get_state()
        world_base = planar(state.base_pose.x, state.base_pose.y, state.base_pose.yaw)
        if mount["parent"] == "base":
            return world_base @ mount["T"]
        if mount["parent"] == "flange":
            if self._flange_in_base is None:
                raise RuntimeError(f"camera {cam!r} is on the flange, but no flange_in_base was given")
            return world_base @ self._flange_in_base(state.arm_q) @ mount["T"]
        raise ValueError(f"camera {cam!r}: unknown parent {mount['parent']!r}")

    # -- time -------------------------------------------------------------------

    def wait_for_next_tick(self, dt: float) -> None:
        """Wall-clock ticks: sleep to the next one (or not at all, when running late)."""
        now = time.monotonic()
        self._next_tick = now + dt if self._next_tick is None or self._next_tick < now - dt else self._next_tick + dt
        time.sleep(max(0.0, self._next_tick - now))
        if self.halted:
            reason, self.halted = self.halted, None
            raise RobotHalted(f"the onboard server held the robot: {reason}")

    def close(self) -> None:
        self._closed.set()
        self._conn.close()

    # -- internals --------------------------------------------------------------

    def _send(self, message: dict) -> None:
        with self._send_lock:
            self._conn.send(pack(message))
            self._last_send = time.monotonic()

    def _beat(self, period_s: float) -> None:
        while not self._closed.wait(period_s / 2):
            if time.monotonic() - self._last_send >= period_s:
                try:
                    self._send({"op": P.HEARTBEAT})
                except ConnectionClosed:
                    return

    def _read(self) -> None:
        try:
            for raw in self._conn:
                message = unpack(raw)
                kind = message.get("type")
                if kind == P.STATE:
                    self._state, self._state_at = message, time.monotonic()
                    self._first_state.set()
                elif kind == P.RGBD:
                    with self._reply_ready:
                        self._replies[message["id"]] = message
                        self._reply_ready.notify_all()
                elif kind == P.HALTED:
                    self.halted = message.get("reason", "")
                    print(f"[real] the robot was held: {self.halted}", flush=True)
                elif kind == P.ERROR:
                    self.errors.append(message.get("message", ""))
                    print(f"[real] onboard error: {message.get('message')}", flush=True)
        except ConnectionClosed:
            pass
        finally:
            self._closed.set()
