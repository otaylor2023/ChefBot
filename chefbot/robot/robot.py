"""Robot: the RobotAPI implemented on top of any Env.

Plans on an inflated occupancy grid, picks standoff poses, runs the base
control loop and the arm skills on one motion thread next to the env.
"""

from __future__ import annotations

from typing import Callable

import math
import threading

import numpy as np

from chefbot.interface.env import Env
from chefbot.interface.fixtures import OPEN, UNKNOWN, FixtureActuator
from chefbot.interface.proposers import GraspProposer, PlacementProposer
from chefbot.interface.robot_api import (
    APPLIANCES,
    ARTICULATE,
    BASE,
    GRASP,
    HANDHELD_TOOLS,
    LIFT_OUT,
    LOWER_INTO,
    STIR,
    RobotAPI,
)
from chefbot.interface.types import Detection, HeldObject, MotionResult, MotionStatus, Pose2D
from chefbot.robot.arm.arm_controller import ArmController, ArmResult
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.config import RobotConfig
from chefbot.robot.motion import MotionManager
from chefbot.robot.nav.astar import AStarPlanner, NoPath, with_yaw
from chefbot.robot.nav.base_controller import BaseController
from chefbot.robot.nav.goal import Reachability, select_standoff, snap_to_free, standoff_candidates
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.safety import base_at_rest, require_base_at_rest, tuck_arm_for_driving
from chefbot.robot.skills.common import ArmStack
from chefbot.robot.skills.pick import run_pick
from chefbot.robot.skills.place import run_place
from chefbot.robot.skills.tool_use import run_lift_out, run_lower_into, run_stir

# Contact is ignored for this long after a motion starts, so a base that was
# stopped against something can still drive away from it.
_CONTACT_GRACE_S = 0.5
_MAX_SETTLE_TICKS = 20
# How far the base may move to get out of a door's way.
_RETREAT_RADIUS = 2.0
_FIXTURE_SETTLE_TICKS = 10

_NEAR_SIDE = (0.05, 0.10)  # m: align retries this far toward the robot when nothing reaches the target


class Robot(RobotAPI):
    def __init__(
        self,
        env: Env,
        grid: OccupancyGrid,
        config: RobotConfig | None = None,
        reachability: Reachability | None = None,
        grasp_proposer: GraspProposer | None = None,
        placement_proposer: PlacementProposer | None = None,
        fixture_actuator: FixtureActuator | None = None,
    ):
        from chefbot.robot.kinematics.collision import ArmCollisionModel
        from chefbot.robot.kinematics.reachability import ReachabilityChecker

        self._env = env
        self._cfg = config or RobotConfig.load()
        self._grid = grid
        # Things that are in the way now but not on the map (open doors), by fixture name.
        self._extra_obstacles: dict[str, np.ndarray] = {}
        self._door_labels: dict[str, str] = {}  # fixture name -> what to call its door in messages
        self._planner = AStarPlanner(grid.inflate(self._cfg.inflation_radius))
        frames = ReachabilityChecker(self._cfg)
        self._reachability = reachability or frames
        self._arm = ArmStack(
            env, self._cfg, frames.ik, ArmController(env, frames.ik, self._cfg), Gripper(env, self._cfg), frames,
            ArmCollisionModel(frames.ik.model),
        )  # fmt: skip
        self._grasp_proposer = grasp_proposer
        self._placement_proposer = placement_proposer
        self._fixture_actuator = fixture_actuator
        self._held_object: HeldObject | None = None
        self._motion = MotionManager(self.get_base_pose)

    # -- RobotAPI ---------------------------------------------------------

    def go_to_pose(self, goal: Pose2D) -> None:
        self._motion.start(lambda preempt: self._run_go_to_pose(goal, preempt))

    def align_to_point(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float).reshape(3)
        self._motion.start(lambda preempt: self._run_align(target, preempt))

    def pick(self, target: Detection) -> None:
        self._motion.start(lambda preempt: self._run_pick(target, preempt))

    def place(self, surface: Detection) -> None:
        self._motion.start(lambda preempt: self._run_place(surface, preempt))

    def actuate_fixture(self, fixture: Detection, open: bool) -> None:
        self._motion.start(lambda preempt: self._run_actuate(fixture, bool(open), preempt))

    def capabilities(self) -> frozenset[str]:
        # Burners and appliances through the fixture actuator; utensils (lower, stir, lift).
        # Not (yet): squeezed / powered gadgets, pouring.
        return frozenset({BASE, GRASP, ARTICULATE, HANDHELD_TOOLS} | ({APPLIANCES} if self._fixture_actuator is not None else set()))

    def use_tool(self, action: str, target: Detection | None = None, seconds: float | None = None) -> None:
        self._motion.start(lambda preempt: self._run_tool(action, target, seconds, preempt))

    def _run_tool(self, action: str, target: Detection | None, seconds: float | None, preempt) -> MotionResult:
        if action == LOWER_INTO and target is not None:
            result = run_lower_into(self._arm, self._placement_proposer, target, self._held, preempt)
        elif action == STIR:
            result = run_stir(self._arm, float(seconds or 20.0), self._held, preempt)
        elif action == LIFT_OUT:
            result = run_lift_out(self._arm, preempt)
        else:
            return self._result(MotionStatus.UNSUPPORTED, f"{action} is not supported by this robot")
        return self._from_arm(result)

    def set_power(self, appliance: Detection, on: bool) -> None:
        self._motion.start(lambda preempt: self._run_power(appliance, bool(on), preempt))

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        return self._motion.wait(timeout_s)

    def get_status(self) -> MotionResult:
        return self._motion.status()

    def stop(self) -> None:
        if not self._motion.stop():
            self._env.set_base_velocity(0.0, 0.0, 0.0)
            self._arm.arm.hold()

    def get_base_pose(self) -> Pose2D:
        return self._env.get_state().base_pose

    @property
    def _held(self) -> HeldObject | None:
        return self._held_object

    @_held.setter
    def _held(self, held: HeldObject | None) -> None:
        self._held_object = held
        if self._arm.collision is not None:
            self._arm.collision.held = None if held is None or held.box_in_tcp is None else np.asarray(held.box_in_tcp)

    def get_held(self) -> HeldObject | None:
        return self._held

    def reset(self) -> None:
        """Stop, and forget the held object and any open doors. For when the
        world has been reset under the robot (a new episode)."""
        self.stop()
        self._arm.arm.hold()
        self._held = None
        self._extra_obstacles.clear()
        self._rebuild_planner()

    # -- Base motions (run on the motion thread) ------------------------------

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
            self._planner, start, target, candidates, self._reachability, preempt.is_set, cfg.standoff_closeness
        )
        # Nothing reaches its middle (a pan at the back of the stove): work on its near side.
        toward = np.array([start.x, start.y]) - np.asarray(target[:2], dtype=float)
        for shift in _NEAR_SIDE if goal is None and np.linalg.norm(toward) > 1e-6 else ():
            if preempt.is_set():
                break
            near = np.asarray(target, dtype=float).copy()
            near[:2] += shift * toward / np.linalg.norm(toward)
            goal = select_standoff(
                self._planner, start, near, standoff_candidates((float(near[0]), float(near[1])), cfg.standoff_min,
                cfg.standoff_max, cfg.standoff_rings, cfg.standoff_angles), self._reachability, preempt.is_set,
                cfg.standoff_closeness,
            )  # fmt: skip
            if goal is not None:
                break
        if preempt.is_set():
            return self._result(MotionStatus.STOPPED, "stopped while choosing a standoff pose")
        if goal is None:
            message = "no collision-free base pose puts the target within arm reach"
            blocking = self._doors_in_the_way(
                lambda planner: select_standoff(planner, start, target, candidates, self._reachability,
                                                closeness=cfg.standoff_closeness) is not None  # fmt: skip
            )
            return self._result(MotionStatus.UNREACHABLE, message + blocking)
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
            blocking = self._doors_in_the_way(lambda planner: _has_path(planner, start, goal))
            return self._result(MotionStatus.NO_PATH, str(exc) + blocking)
        if start is not pose:
            # The base sits inside the inflated margin: leave it in a straight line first.
            points = [(pose.x, pose.y)] + points
        # Interlock: the arm is tucked (carrying, if it holds something) before the base moves.
        tucked = tuck_arm_for_driving(self._arm.arm, self._cfg, self._held, preempt)
        if not tucked.ok:
            return self._from_arm(tucked)
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
                if state.arm_in_contact and elapsed > _CONTACT_GRACE_S:
                    status, message = MotionStatus.BLOCKED, "the arm touched something while driving"
                    break
                if self._held is not None and not self._arm.gripper.is_holding():
                    label, self._held = self._held.label, None
                    status = MotionStatus.GRASP_FAILED
                    message = f"the {label} slipped out of the gripper while driving; it may have dropped"
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
        self._settle(dt)
        return self._result(status, message)

    def _settle(self, dt: float) -> None:
        """Tick with a zero command until the base has come to rest, so the
        reported pose is where the robot actually ends up."""
        for _ in range(_MAX_SETTLE_TICKS):
            self._env.wait_for_next_tick(dt)
            if base_at_rest(self._env):
                break

    # -- Arm skills (run on the motion thread) --------------------------------

    def _run_pick(self, target: Detection, preempt: threading.Event) -> MotionResult:
        result, self._held = run_pick(self._arm, self._grasp_proposer, target, self._held, preempt)
        return self._from_arm(result)

    def _run_place(self, surface: Detection, preempt: threading.Event) -> MotionResult:
        result, self._held = run_place(self._arm, self._placement_proposer, surface, self._held, preempt)
        return self._from_arm(result)

    def _run_power(self, appliance: Detection, on: bool, preempt: threading.Event) -> MotionResult:
        """Turn a burner's knob (or an appliance) on / off: scripted, like doors."""
        actuator = self._fixture_actuator
        if actuator is None or actuator.get_power(appliance) == UNKNOWN:
            return self._result(MotionStatus.UNSUPPORTED, f"the {appliance.label} cannot be turned on or off")
        moving = require_base_at_rest(self._env)
        if not moving.ok:
            return self._from_arm(moving)
        state = "on" if on else "off"
        if actuator.get_power(appliance) == state:
            return self._result(MotionStatus.SUCCESS, f"the {appliance.label} is already {state}", scripted=True)
        within_reach = getattr(self._reachability, "can_touch", self._reachability.is_reachable)
        if not within_reach(self.get_base_pose(), appliance.position_world):
            return self._result(MotionStatus.UNREACHABLE, f"the {appliance.label} is out of arm reach from here; align to it first")
        result = actuator.set_power(appliance, on)
        for _ in range(_FIXTURE_SETTLE_TICKS):
            self._env.wait_for_next_tick(self._cfg.control_dt)
        if not result.success:
            return self._result(MotionStatus.UNSUPPORTED, result.message, scripted=result.scripted)
        return self._result(MotionStatus.SUCCESS, result.message, scripted=result.scripted)

    def _run_actuate(self, fixture: Detection, open: bool, preempt: threading.Event) -> MotionResult:
        actuator, verb = self._fixture_actuator, "open" if open else "close"
        state = actuator.get_state(fixture) if actuator is not None else UNKNOWN
        if state == UNKNOWN:
            return self._result(MotionStatus.UNSUPPORTED, f"{fixture.label} is not an openable fixture")
        moving = require_base_at_rest(self._env)
        if not moving.ok:
            return self._from_arm(moving)
        if (state == OPEN) == open:
            return self._result(MotionStatus.SUCCESS, f"the {fixture.label} is already {state}", scripted=True)
        # The gripper has to be able to get to it; how it would take hold is not planned yet (scripted).
        within_reach = getattr(self._reachability, "can_touch", self._reachability.is_reachable)
        if not within_reach(self.get_base_pose(), fixture.position_world):
            return self._result(
                MotionStatus.UNREACHABLE, f"the {fixture.label} is out of arm reach from here; align to it first"
            )

        # A door swings through the floor space in front of it: step out of its way first.
        swept = actuator.swept_xy(fixture, open)
        if swept is not None and len(swept) and self._inside(swept, self.get_base_pose()):
            goal = self._retreat_goal(swept, fixture.position_world)
            if goal is None:
                return self._result(MotionStatus.BLOCKED, f"no room to step out of the way to {verb} the {fixture.label}")
            moved = self._plan_and_drive(goal, preempt)
            if moved.status != MotionStatus.SUCCESS:
                return moved
        else:
            result = self._arm.arm.move_joints(self._arm.ready, preempt)
            if not result.ok:
                return self._from_arm(result)

        actuation = actuator.set_state(fixture, open)
        for _ in range(_FIXTURE_SETTLE_TICKS):
            self._env.wait_for_next_tick(self._cfg.control_dt)
        if not actuation.success:
            return self._result(MotionStatus.UNSUPPORTED, actuation.message, scripted=actuation.scripted)
        if actuation.fixture:
            self._door_labels[actuation.fixture] = fixture.label.removesuffix(" handle")
            if actuation.obstacle_xy is not None and len(actuation.obstacle_xy):
                self._extra_obstacles[actuation.fixture] = np.asarray(actuation.obstacle_xy, dtype=float)
            else:
                self._extra_obstacles.pop(actuation.fixture, None)
            self._rebuild_planner()
        message = actuation.message or f"the {fixture.label} is now {'open' if open else 'closed'}"
        return self._result(MotionStatus.SUCCESS, message, scripted=actuation.scripted)

    # -- obstacles that are not on the map --------------------------------------

    def _grid_with(self, extra: list[np.ndarray]) -> OccupancyGrid:
        occupied = self._grid.occupied.copy()
        for points in extra:
            for x, y in points:
                cell = self._grid.world_to_cell(float(x), float(y))
                if self._grid.in_bounds(cell):
                    occupied[cell] = True
        return OccupancyGrid(occupied, self._grid.resolution, self._grid.origin)

    def _doors_in_the_way(self, works: Callable[[AStarPlanner], bool]) -> str:
        """If closing doors this robot opened would make `works(planner)` true: a hint naming them."""
        if not self._extra_obstacles:
            return ""
        grid = self._grid.inflate(self._cfg.inflation_radius)
        if not works(AStarPlanner(grid)):
            return ""
        names = [self._door_labels.get(name, "door") for name in self._extra_obstacles]
        return f"; the open {' and '.join(dict.fromkeys(names))} door is in the way: close it first"

    def _rebuild_planner(self) -> None:
        grid = self._grid_with(list(self._extra_obstacles.values()))
        self._planner = AStarPlanner(grid.inflate(self._cfg.inflation_radius))

    def _inside(self, points: np.ndarray, pose: Pose2D) -> bool:
        """True if the base footprint at `pose` overlaps any of the xy `points`."""
        distances = np.linalg.norm(np.asarray(points, dtype=float) - [pose.x, pose.y], axis=1)
        return bool(distances.min() <= self._cfg.inflation_radius)

    def _retreat_goal(self, swept: np.ndarray, look_at: np.ndarray) -> Pose2D | None:
        """The cheapest-to-reach base pose that is clear of `swept`, facing `look_at`."""
        pose = self.get_base_pose()
        start = self._free_start(pose)
        if start is None:
            return None
        clear = self._grid_with([*self._extra_obstacles.values(), swept]).inflate(self._cfg.inflation_radius)
        now = self._planner.grid
        row, col = now.world_to_cell(pose.x, pose.y)
        reach = math.ceil(_RETREAT_RADIUS / now.resolution)
        cells = {
            (r, c)
            for r in range(row - reach, row + reach + 1)
            for c in range(col - reach, col + reach + 1)
            if now.is_free_cell((r, c)) and clear.is_free_cell((r, c))
        }
        costs = self._planner.cost_to((start.x, start.y), cells)
        if not costs:
            return None
        x, y = now.cell_to_world(min(costs, key=costs.get))
        return Pose2D(x, y, math.atan2(look_at[1] - y, look_at[0] - x))

    # -- results ------------------------------------------------------------------

    def _result(self, status: str, message: str, **data) -> MotionResult:
        return MotionResult(status, self.get_base_pose(), message, dict(data))

    def _from_arm(self, result: ArmResult) -> MotionResult:
        return self._result(result.status, result.message)


def _has_path(planner: AStarPlanner, start: Pose2D, goal: Pose2D) -> bool:
    try:
        planner.plan_xy((start.x, start.y), (goal.x, goal.y))
    except NoPath:
        return False
    return True
