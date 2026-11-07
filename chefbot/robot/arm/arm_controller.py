"""ArmController: joint-space moves and straight-line Cartesian moves.

Turns goals into a stream of targets for the env, one per control tick: joint
targets, or, on an arm whose own controller moves to poses (a real arm's
driver, Env.set_arm_pose_target), the flange poses along a planned line. Lines
are still planned with our IK, which checks they are reachable. Every tick it
checks for a stop request and for unexpected contact, and it never leaves the
arm chasing a target after it returns. A move whose path would take the arm into
the robot's own body (its constraints, embodiments/constraints.py: the pole,
the camera clamp, the base's plate) is refused before it starts.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.types import MotionStatus
from chefbot.robot.arm.ik import ArmIK, rotation_angle
from chefbot.robot.config import RobotConfig


@dataclass
class ArmResult:
    status: str  # MotionStatus: SUCCESS, STOPPED, COLLISION, UNREACHABLE, TIMEOUT
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == MotionStatus.SUCCESS


OK = ArmResult(MotionStatus.SUCCESS)
_LEAVE_CONTACT_S = 1.0


def smoothstep(t: float) -> float:
    """Quintic ease from 0 to 1 with zero velocity and acceleration at both ends."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def move_duration(distance: float, max_vel: float, max_acc: float) -> float:
    """Shortest duration of a smoothstep move over `distance` within the limits.
    (Peak velocity of the profile is 1.875 d/T, peak acceleration 5.774 d/T^2.)"""
    return max(1.875 * distance / max_vel, math.sqrt(5.774 * distance / max_acc))


def slerp_pose(start: np.ndarray, goal: np.ndarray, s: float) -> np.ndarray:
    """Pose a fraction `s` of the way from `start` to `goal`: position lerp, orientation slerp."""
    pose = np.eye(4)
    pose[:3, 3] = (1.0 - s) * start[:3, 3] + s * goal[:3, 3]
    relative = start[:3, :3].T @ goal[:3, :3]
    angle = math.acos(min(1.0, max(-1.0, (np.trace(relative) - 1.0) / 2.0)))
    if angle < 1e-6:
        pose[:3, :3] = goal[:3, :3]
        return pose
    axis = np.array(
        [relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]
    ) / (2.0 * math.sin(angle))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    theta = s * angle
    pose[:3, :3] = start[:3, :3] @ (np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * k @ k)
    return pose


class ArmController:
    def __init__(self, env: Env, ik: ArmIK, config: RobotConfig, constraints: list = (), hand_m: float = 0.0):
        self._env = env
        self._constraints, self._hand_m = list(constraints), hand_m  # the robot's body; none: nothing refused
        self._ik = ik
        self._cfg = config
        self._target: np.ndarray | None = None  # last joint target sent
        # The TCP in the flange frame, to send flange poses for TCP poses.
        self._flange_to_tcp = np.eye(4)
        c, s_ = math.cos(config.tcp_yaw), math.sin(config.tcp_yaw)
        self._flange_to_tcp[:3, :3] = [[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]
        self._flange_to_tcp[:3, 3] = config.tcp_xyz
        self._moves_poses: bool | None = None  # whether the env takes pose targets; asked on the first line

    # -- planning (no motion) ---------------------------------------------------

    def current_target(self) -> np.ndarray:
        """Where the arm has been told to be: the last target, or its measured pose."""
        measured = np.asarray(self._env.get_state().arm_q, dtype=float)
        # No target yet, or the arm is nowhere near the last one (it was moved or
        # reset from outside): start from where it actually is.
        if self._target is None or np.abs(measured - self._target).max() > self._cfg.tracking_tolerance / 2:
            self._target = measured.copy()
        return self._target.copy()

    def plan_line(self, start_q: np.ndarray, goal: np.ndarray) -> list[np.ndarray] | None:
        """Joint waypoints taking the TCP in a straight line from where `start_q`
        puts it to the pose `goal` (arm base frame), sampled every `cart_step`.
        None if any sample has no IK solution or the solution jumps."""
        cfg = self._cfg
        start = self._ik.fk(start_q)
        length = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        steps = max(1, math.ceil(length / cfg.cart_step))
        waypoints = [np.asarray(start_q, dtype=float)]
        for i in range(1, steps + 1):
            q = self._ik.solve_pose(slerp_pose(start, goal, i / steps), seed=waypoints[-1])
            if q is None or np.abs(q - waypoints[-1]).max() > cfg.max_joint_jump:
                return None
            waypoints.append(q)
        return waypoints

    # -- motion -----------------------------------------------------------------

    def move_joints(self, goal: np.ndarray, preempt: threading.Event, expect_contact: bool = False) -> ArmResult:
        """Interpolate from the current target to `goal` within the joint limits on speed and acceleration."""
        cfg = self._cfg
        goal = np.asarray(goal, dtype=float)
        if not self._ik.within_limits(goal, margin=1e-6):
            return ArmResult(MotionStatus.UNREACHABLE, "joint goal is outside the joint limits")
        start = self.current_target()
        blocked = self._into_body([start, goal])
        if blocked is not None:
            return blocked
        distance = float(np.abs(goal - start).max())
        duration = move_duration(distance, cfg.max_joint_vel, cfg.max_joint_acc)
        return self._run(lambda s: start + s * (goal - start), duration, goal, preempt, expect_contact)

    def follow(
        self, waypoints: list[np.ndarray], preempt: threading.Event, expect_contact: bool = False, speed: float = 1.0
    ) -> ArmResult:
        """Execute joint waypoints from `plan_line` at the Cartesian speed limit
        (times `speed`: below 1 for gentle moves, like pouring)."""
        cfg = self._cfg
        if len(waypoints) < 2:
            return OK
        blocked = self._into_body(waypoints)
        if blocked is not None:
            return blocked
        points = np.asarray(waypoints, dtype=float)
        length = float(np.linalg.norm(self._ik.fk(points[-1])[:3, 3] - self._ik.fk(points[0])[:3, 3]))
        joint_distance = float(np.abs(np.diff(points, axis=0)).sum(axis=0).max())
        duration = max(
            move_duration(length, cfg.cart_speed, cfg.cart_acc),
            move_duration(joint_distance, cfg.max_joint_vel, cfg.max_joint_acc),
        ) / speed

        def along(s: float) -> np.ndarray:
            position = s * (len(points) - 1)
            i = min(int(position), len(points) - 2)
            return points[i] + (position - i) * (points[i + 1] - points[i])

        if self._env_moves_poses():
            return self._run_poses(lambda s: self._ik.fk(along(s)), duration, self._ik.fk(points[-1]), preempt, expect_contact)
        return self._run(along, duration, points[-1], preempt, expect_contact)

    def move_line(self, goal: np.ndarray, preempt: threading.Event, expect_contact: bool = False) -> ArmResult:
        """Straight-line TCP move from the current target to the pose `goal` (arm base frame)."""
        waypoints = self.plan_line(self.current_target(), goal)
        if waypoints is None:
            return ArmResult(MotionStatus.UNREACHABLE, "no_ik: the straight-line path has no joint solution")
        return self.follow(waypoints, preempt, expect_contact)

    def hold(self) -> None:
        """Stop where the arm is now."""
        self._send(np.asarray(self._env.get_state().arm_q, dtype=float))

    def at(self, q: np.ndarray) -> bool:
        return bool(np.abs(self._env.get_state().arm_q - np.asarray(q)).max() <= self._cfg.joint_tolerance)

    # -- internals --------------------------------------------------------------

    def _into_body(self, waypoints) -> ArmResult | None:
        """UNREACHABLE if the path goes into the robot's own body, else None."""
        if not self._constraints:
            return None
        from chefbot.embodiments.constraints import path_violation

        where = path_violation(self._constraints, self._hand_m, waypoints)
        if where is None:
            return None
        return ArmResult(MotionStatus.UNREACHABLE, f"the arm would go into {where}")

    def _env_moves_poses(self) -> bool:
        """Whether the env's arm takes pose targets. Asked once, with the pose the arm is at."""
        if self._moves_poses is None:
            here = self._ik.fk(np.asarray(self._env.get_state().arm_q, dtype=float))
            self._moves_poses = bool(self._env.set_arm_pose_target(self._flange(here)))
        return self._moves_poses

    def _flange(self, tcp_pose: np.ndarray) -> np.ndarray:
        return tcp_pose @ np.linalg.inv(self._flange_to_tcp)

    def _run_poses(self, tcp_at, duration: float, goal: np.ndarray, preempt, expect_contact: bool) -> ArmResult:
        """`_run` for an arm that moves to poses itself: TCP poses along the line, tracking checked in
        Cartesian space (its controller picks its own joint angles, so ours are only a prediction)."""
        env, cfg = self._env, self._cfg
        dt = cfg.control_dt
        ticks = max(1, math.ceil(duration / dt))
        settle_ticks = math.ceil(cfg.arm_settle_s / dt)
        leaving = bool(env.get_state().arm_in_contact)
        commanded: np.ndarray | None = None
        for tick in range(1, ticks + settle_ticks + 1):
            state = env.get_state()
            if preempt.is_set():
                self.hold()
                return ArmResult(MotionStatus.STOPPED, "motion was stopped")
            leaving = leaving and state.arm_in_contact and tick * dt <= _LEAVE_CONTACT_S
            if state.arm_in_contact and not expect_contact and not leaving:
                self.hold()
                return ArmResult(MotionStatus.COLLISION, "the arm touched something")
            tcp = self._ik.fk(np.asarray(state.arm_q, dtype=float))
            if commanded is not None and np.linalg.norm(tcp[:3, 3] - commanded[:3, 3]) > cfg.cart_tracking_tolerance:
                self.hold()
                return ArmResult(MotionStatus.COLLISION, "the arm is blocked and cannot follow its path")
            if (
                tick > ticks
                and np.linalg.norm(tcp[:3, 3] - goal[:3, 3]) <= cfg.cart_goal_tolerance
                and rotation_angle(tcp[:3, :3].T @ goal[:3, :3]) <= cfg.cart_goal_rot_tolerance
            ):
                self._target = np.asarray(state.arm_q, dtype=float).copy()
                return OK
            commanded = tcp_at(smoothstep(tick / ticks))
            env.set_arm_pose_target(self._flange(commanded))
            env.wait_for_next_tick(dt)
        self.hold()
        return ArmResult(MotionStatus.TIMEOUT, "the arm did not reach its target")

    def _send(self, q: np.ndarray) -> None:
        self._target = np.asarray(q, dtype=float).copy()
        self._env.set_arm_joint_targets(self._target)

    def _run(self, target_at, duration: float, goal: np.ndarray, preempt, expect_contact: bool) -> ArmResult:
        env, cfg = self._env, self._cfg
        dt = cfg.control_dt
        ticks = max(1, math.ceil(duration / dt))
        settle_ticks = math.ceil(cfg.arm_settle_s / dt)
        # An arm that starts out touching something (it stopped there after a
        # collision) is given a moment to pull clear before contact counts again.
        leaving = bool(env.get_state().arm_in_contact)
        self.current_target()  # resynchronize before tracking is checked
        for tick in range(1, ticks + settle_ticks + 1):
            state = env.get_state()
            if preempt.is_set():
                self.hold()
                return ArmResult(MotionStatus.STOPPED, "motion was stopped")
            leaving = leaving and state.arm_in_contact and tick * dt <= _LEAVE_CONTACT_S
            if state.arm_in_contact and not expect_contact and not leaving:
                self.hold()
                return ArmResult(MotionStatus.COLLISION, "the arm touched something")
            if self._target is not None and np.abs(state.arm_q - self._target).max() > cfg.tracking_tolerance:
                self.hold()
                return ArmResult(MotionStatus.COLLISION, "the arm is blocked and cannot follow its path")
            if tick > ticks and np.abs(state.arm_q - goal).max() <= cfg.joint_tolerance:
                return OK
            self._send(target_at(smoothstep(tick / ticks)))
            env.wait_for_next_tick(dt)
        self.hold()
        return ArmResult(MotionStatus.TIMEOUT, "the arm did not reach its target")
