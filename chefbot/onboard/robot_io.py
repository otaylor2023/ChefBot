"""The robot's drivers through Redis: one state read, and the commands the server passes on.

State is one MGET per read. Commands that repeat (the gripper, which controller is active, the
base speed scale) are written only when they change: the local Redis has been swamped before by
drivers and clients writing every tick.
"""

from __future__ import annotations

import json
import time

import numpy as np

from chefbot.onboard import keys as K
from chefbot.onboard.frames import BaseFrames
from chefbot.onboard.mocap import MocapFreshness

_STATE_KEYS = (
    K.JOINT_POSITIONS, K.JOINT_VELOCITIES, K.GRIPPER_CURRENT_WIDTH, K.GRIPPER_GRASP_SUCCESS, K.GRIPPER_READY,
    K.GRIPPER_MAX_WIDTH, K.BASE_POSE, K.BASE_VELOCITY, K.MOCAP_POSITION, K.MOCAP_ORIENTATION, K.MOCAP_VALID,
    K.ACTIVE_CONTROLLER, K.CARTESIAN_CURRENT_POSITION, K.CARTESIAN_CURRENT_ORIENTATION,
)  # fmt: skip


def _text(raw) -> str | None:
    if raw is None:
        return None
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _array(raw) -> np.ndarray | None:
    text = _text(raw)
    if not text:
        return None
    try:
        return np.asarray(json.loads(text), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _number(raw) -> float | None:
    try:
        return float(_text(raw))
    except (TypeError, ValueError):
        return None


def _flag(raw) -> bool:
    return (_text(raw) or "").strip().lower() in ("1", "true", "yes", "on")


def _list(values) -> str:
    return json.dumps([float(v) for v in values])


class RedisRobot:
    """`redis` is a client with get / set / mget (redis.Redis, or a fake in tests)."""

    def __init__(self, redis, frames: BaseFrames, gripper: dict, base_max_vel: tuple[float, float],
                 mocap: MocapFreshness | None = None):  # fmt: skip
        self.r = redis
        self.frames = frames
        self.mocap = mocap or MocapFreshness()
        self.gripper_speed = float(gripper["speed"])
        self.gripper_force = float(gripper["force"])
        self.base_max_vel = base_max_vel  # the base driver's --max-vel-xy, --max-vel-yaw
        # The arm driver's joints change every sample (sensor noise alone): frozen, the driver is gone (its last
        # values stay in Redis). When they last changed:
        self._arm_raw, self._arm_changed = None, time.monotonic()
        self._written: dict[str, str] = {}  # last value of each write-on-change key
        self.gripper_closed: bool | None = None

    # -- state ------------------------------------------------------------------

    def read_state(self) -> dict:
        v = dict(zip(_STATE_KEYS, self.r.mget(_STATE_KEYS)))
        if v[K.JOINT_POSITIONS] != self._arm_raw:
            self._arm_raw, self._arm_changed = v[K.JOINT_POSITIONS], time.monotonic()
        odom = _array(v[K.BASE_POSE])
        mocap_valid = _flag(v[K.MOCAP_VALID])
        mocap_age = self.mocap.update(v[K.MOCAP_POSITION], v[K.MOCAP_ORIENTATION])
        mocap_fresh = self.mocap.is_fresh(mocap_age)
        position, quat = _array(v[K.MOCAP_POSITION]), _array(v[K.MOCAP_ORIENTATION])
        # The base pose from mocap alone (no odometry needed), only when the publisher is live and tracking.
        mocap_pose = None
        if mocap_valid and mocap_fresh and position is not None and quat is not None:
            mocap_pose = self.frames.base_from_mocap(position, quat)
            if odom is not None:
                self.frames.tie(mocap_pose, odom[:3])
        base_velocity = _array(v[K.BASE_VELOCITY])
        cart_position, cart_orientation = _array(v[K.CARTESIAN_CURRENT_POSITION]), _array(v[K.CARTESIAN_CURRENT_ORIENTATION])
        return {
            "base_pose": None if odom is None else self.frames.to_world(odom[:3]),
            "base_velocity": None if base_velocity is None else self.frames.velocity_to_world(base_velocity[:3]),
            "localized": self.frames.tied,  # False: never seen mocap; the world frame is odometry
            "mocap_valid": mocap_valid,  # the publisher's tracking flag, as stored (may be stale)
            "mocap_fresh": mocap_fresh,  # the pose changed within mocap.stale_after_s
            "mocap_age_s": mocap_age,  # seconds since the pose last changed; None: not since the server started
            "mocap_pose": mocap_pose,  # (x, y, yaw) world, from mocap alone; None unless valid and fresh
            "arm_q": _array(v[K.JOINT_POSITIONS]),
            "arm_qd": _array(v[K.JOINT_VELOCITIES]),
            "arm_frozen_s": round(time.monotonic() - self._arm_changed, 3),  # > ~0.2 s: the arm driver isn't publishing
            "gripper_width": _number(v[K.GRIPPER_CURRENT_WIDTH]),
            "gripper_max_width": _number(v[K.GRIPPER_MAX_WIDTH]),
            "gripper_ready": _text(v[K.GRIPPER_READY]),
            "gripper_closed_cmd": bool(self.gripper_closed),
            "grasp_success": _text(v[K.GRIPPER_GRASP_SUCCESS]),
            "controller": _text(v[K.ACTIVE_CONTROLLER]),
            # The arm controller's own pose of the point it moves (arm base frame), for checking frames.
            "driver_position": cart_position,
            "driver_orientation": None if cart_orientation is None else cart_orientation.reshape(3, 3),
        }

    # -- commands ---------------------------------------------------------------

    def arm_joints(self, q) -> None:
        """Joint targets to the joint controller (goal first, then the switch, as ZitiBot does)."""
        q = np.asarray(q, dtype=float).reshape(-1)
        self.r.set(K.JOINT_GOAL_POSITION, _list(q))
        if self._written.get(K.ACTIVE_CONTROLLER) != K.JOINT_CONTROLLER:
            zeros = _list(np.zeros_like(q))
            self.r.set(K.JOINT_GOAL_VELOCITY, zeros)
            self.r.set(K.JOINT_GOAL_ACCELERATION, zeros)
        self._set_on_change(K.ACTIVE_CONTROLLER, K.JOINT_CONTROLLER)

    def arm_pose(self, position, orientation) -> None:
        """A pose of the Cartesian controller's point (arm base frame) to the Cartesian controller."""
        self.r.set(K.CARTESIAN_GOAL_POSITION, _list(np.asarray(position, dtype=float).reshape(3)))
        self.r.set(K.CARTESIAN_GOAL_ORIENTATION, json.dumps(np.asarray(orientation, dtype=float).reshape(3, 3).tolist()))
        self._set_on_change(K.ACTIVE_CONTROLLER, K.CARTESIAN_CONTROLLER)

    def gripper(self, closed: bool) -> None:
        """Closed: a grasp; open: a move to full width; at the configured speed and force, when it changes."""
        if closed == self.gripper_closed:
            return
        if closed:
            self.gripper_grasp(self.gripper_speed, self.gripper_force)
        else:
            self.gripper_move(_number(self.r.get(K.GRIPPER_MAX_WIDTH)) or 0.08, self.gripper_speed)

    def gripper_move(self, width: float, speed: float) -> None:
        """libfranka's move(width, speed) (the driver retries 1 cm wider when something stops the fingers)."""
        self.gripper_closed = False
        self._gripper_command(K.GRIPPER_MOVE, width, speed, self.gripper_force)

    def gripper_grasp(self, speed: float, force: float) -> None:
        """libfranka's grasp(0, speed, force, 1.0, 1.0): close until the fingers stop, then hold `force`; the driver
        reports grasp_success."""
        self.gripper_closed = True
        self.r.set(K.GRIPPER_GRASP_SUCCESS, K.GRIPPER_PENDING)
        self._gripper_command(K.GRIPPER_GRASP, 0.0, speed, force)

    def _gripper_command(self, mode: str, width: float, speed: float, force: float) -> None:
        self.r.set(K.GRIPPER_DESIRED_WIDTH, str(min(max(float(width), 0.0), 0.08)))
        self.r.set(K.GRIPPER_DESIRED_SPEED, str(min(max(float(speed), 0.01), 0.1)))
        self.r.set(K.GRIPPER_DESIRED_FORCE, str(min(max(float(force), 0.1), 70.0)))
        self.r.set(K.GRIPPER_MODE, mode)

    def base_target(self, pose_world, max_vel) -> None:
        """Drive to a world pose within `max_vel` (never faster than the driver's own limits)."""
        scale = min(float(max_vel[0]) / self.base_max_vel[0], float(max_vel[2]) / self.base_max_vel[1], 1.0)
        self._set_on_change(K.BASE_MAX_VEL_SCALE, f"{max(scale, 0.1):.3f}")
        x, y, yaw = self.frames.to_odom(pose_world)
        self.r.set(K.BASE_DESIRED_POSE, f"[{x}, {y}, {yaw}]")
        self._written.pop(K.BASE_STOP, None)

    def base_hold(self) -> None:
        """Decelerate and hold where it is (the driver resets the flag once it has)."""
        self._set_on_change(K.BASE_STOP, "stop")

    def halt(self) -> None:
        """Hold the arm where it is and stop the base; the gripper stays as it is."""
        q = _array(self.r.get(K.JOINT_POSITIONS))
        if q is not None and q.size:
            self._written.pop(K.ACTIVE_CONTROLLER, None)
            self.arm_joints(q)
        self._written.pop(K.BASE_STOP, None)
        self.base_hold()

    def wait_gripper_ready(self, timeout_s: float = 30.0) -> bool:
        """True once the gripper driver has homed (or does not report it)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready = _text(self.r.get(K.GRIPPER_READY))
            if ready is None or ready.strip() == "1":
                return True
            time.sleep(0.1)
        return False

    def _set_on_change(self, key: str, value: str) -> None:
        if self._written.get(key) != value:
            self.r.set(key, value)
            self._written[key] = value
