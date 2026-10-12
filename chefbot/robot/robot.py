"""Robot: the RobotAPI implemented on top of any Env.

Plans on an inflated occupancy grid, picks standoff poses, and runs the base
control loop on a motion thread next to the env.
"""

from __future__ import annotations

import threading

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import MotionResult, MotionStatus, Pose2D
from chefbot.robot.config import RobotConfig
from chefbot.robot.motion import MotionManager
from chefbot.robot.nav.astar import AStarPlanner, NoPath, with_yaw
from chefbot.robot.nav.base_controller import BaseController
from chefbot.robot.nav.goal import Reachability, select_standoff, snap_to_free, standoff_candidates
from chefbot.robot.nav.grid import OccupancyGrid

# Contact is ignored for this long after a motion starts, so a base that was
# stopped against something can still drive away from it.
_CONTACT_GRACE_S = 0.5


class Robot(RobotAPI):
    def __init__(
        self,
        env: Env,
        grid: OccupancyGrid,
        config: RobotConfig | None = None,
        reachability: Reachability | None = None,
    ):
        self._env = env
        self._cfg = config or RobotConfig.load()
        self._planner = AStarPlanner(grid.inflate(self._cfg.inflation_radius))
        if reachability is None:
            from chefbot.robot.kinematics.reachability import ReachabilityChecker

            reachability = ReachabilityChecker(self._cfg)
        self._reachability = reachability
        self._motion = MotionManager(self.get_base_pose)

    # -- RobotAPI ---------------------------------------------------------

    def go_to_pose(self, goal: Pose2D) -> None:
        self._motion.start(lambda preempt: self._run_go_to_pose(goal, preempt))

    def align_to_point(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float).reshape(3)
        self._motion.start(lambda preempt: self._run_align(target, preempt))

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        return self._motion.wait(timeout_s)

    def get_status(self) -> MotionResult:
        return self._motion.status()

    def stop_base(self) -> None:
        if not self._motion.stop():
            self._env.set_base_velocity(0.0, 0.0, 0.0)

    def get_base_pose(self) -> Pose2D:
        return self._env.get_state().base_pose

    # -- Motions (run on the motion thread) --------------------------------

    def _run_go_to_pose(self, goal: Pose2D, preempt: threading.Event) -> MotionResult:
        grid, cfg = self._planner.grid, self._cfg
        snapped = snap_to_free(grid, goal, cfg.snap_radius)
        if snapped is None:
            return self._result(
                MotionStatus.NO_PATH,
                f"goal is in collision and no free space within {cfg.snap_radius:.2f} m",
            )
        return self._plan_and_drive(snapped, preempt)

    def _run_align(self, target: np.ndarray, preempt: threading.Event) -> MotionResult:
        cfg = self._cfg
        start = self._free_start(self.get_base_pose())
        if start is None:
            return self._result(MotionStatus.NO_PATH, "base is in collision on the map")
        candidates = standoff_candidates(
            (float(target[0]), float(target[1])),
            cfg.standoff_min,
            cfg.standoff_max,
            cfg.standoff_rings,
            cfg.standoff_angles,
        )
        goal = select_standoff(
            self._planner, start, target, candidates, self._reachability, preempt.is_set
        )
        if preempt.is_set():
            return self._result(MotionStatus.STOPPED, "stopped while choosing a standoff pose")
        if goal is None:
            return self._result(
                MotionStatus.UNREACHABLE,
                "no collision-free base pose puts the target within arm reach",
            )
        return self._plan_and_drive(goal, preempt)

    def _free_start(self, pose: Pose2D) -> Pose2D | None:
        return snap_to_free(self._planner.grid, pose, self._cfg.snap_radius)

    def _plan_and_drive(self, goal: Pose2D, preempt: threading.Event) -> MotionResult:
        pose = self.get_base_pose()
        start = self._free_start(pose)
        if start is None:
            return self._result(MotionStatus.NO_PATH, "base is in collision on the map")
        try:
            points = self._planner.plan_xy((start.x, start.y), (goal.x, goal.y))
        except NoPath as exc:
            return self._result(MotionStatus.NO_PATH, str(exc))
        if start is not pose:
            # The base sits inside the inflated margin: leave it in a straight line first.
            points = [(pose.x, pose.y)] + points
        return self._drive(with_yaw(points, pose.yaw, goal.yaw), preempt)

    def _drive(self, path: list[Pose2D], preempt: threading.Event) -> MotionResult:
        env, cfg = self._env, self._cfg
        controller = BaseController(path, cfg)
        dt = cfg.control_dt
        start_time: float | None = None
        try:
            while True:
                if preempt.is_set():
                    status, message = MotionStatus.STOPPED, "motion was stopped"
                    break
                state = env.get_state()
                if start_time is None:
                    start_time = state.time
                elapsed = state.time - start_time
                if state.base_in_contact and elapsed > _CONTACT_GRACE_S:
                    status, message = MotionStatus.BLOCKED, "base is in contact with an obstacle"
                    break
                if controller.at_goal(state.base_pose):
                    status, message = MotionStatus.SUCCESS, "reached the goal"
                    break
                if elapsed > cfg.motion_timeout_s:
                    status, message = MotionStatus.TIMEOUT, "motion timed out"
                    break
                command = controller.step(state.base_pose, dt)
                if controller.deviation > cfg.deviation_threshold:
                    status = MotionStatus.BLOCKED
                    message = f"base is {controller.deviation:.2f} m off the planned path"
                    break
                env.set_base_velocity(*command, frame="world")
                env.wait_for_next_tick(dt)
        finally:
            # Velocity is zeroed on every exit, including exceptions.
            env.set_base_velocity(0.0, 0.0, 0.0)
        env.wait_for_next_tick(dt)  # let the stop take effect before reporting the pose
        return self._result(status, message)

    def _result(self, status: str, message: str) -> MotionResult:
        return MotionResult(status, self.get_base_pose(), message)
