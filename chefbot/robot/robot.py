"""Robot: the RobotAPI implemented on top of any Env.

Plans on an inflated occupancy grid, picks standoff poses, runs the base
control loop and the arm skills on one motion thread next to the env.
"""

from __future__ import annotations

from typing import Callable

import math
import threading
import time

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
    POUR,
    POUR_INTO,
    STIR,
    RobotAPI,
)
from chefbot.interface.types import Detection, HeldObject, MotionResult, MotionStatus, Pose2D, wrap_angle
from chefbot.interface.workspace import Footprint
from chefbot.robot.arm.arm_controller import ArmController, ArmResult
from chefbot.robot.arm.gripper import Gripper
from chefbot.robot.config import RobotConfig
from chefbot.robot.localization import Localizer
from chefbot.robot.motion import MotionManager
from chefbot.robot.nav.astar import AStarPlanner, NoPath, with_yaw
from chefbot.robot.nav.base_controller import BaseController
from chefbot.robot.nav.goal import Reachability, select_standoff, snap_to_free, standoff_candidates
from chefbot.robot.nav.footprint import FootprintSpace
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.nav.heading_plan import PathStyle, escape_start, headings, plan_at_headings, subdivide
from chefbot.robot.safety import BaseGuard, base_at_rest, require_base_at_rest, tuck_arm_for_driving
from chefbot.robot.skills.common import ArmStack
from chefbot.robot.skills.pick import run_pick
from chefbot.robot.skills.place import run_place
from chefbot.robot.skills.pour import run_pour
from chefbot.robot.skills.tool_use import run_lift_out, run_lower_into, run_stir

# Contact is ignored for this long after a motion starts, so a base that was
# stopped against something can still drive away from it.
_CONTACT_GRACE_S = 0.5
_MAX_SETTLE_TICKS = 20
# How far the base may move to get out of a door's way.
_RETREAT_RADIUS = 2.0
_FIXTURE_SETTLE_TICKS = 10

_NEAR_SIDE = (0.05, 0.10)
_ARM_MOVED_RAD = 0.05  # with tuck_arm=False: a joint moving more than this while the base drives stops it  # m: align retries this far toward the robot when nothing reaches the target


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
        footprint_below: Callable | None = None,
        stop_model=None,
        base_accel: float = 0.1,
        centre_limit: OccupancyGrid | None = None,
        tuck_arm: bool = True,
        drive_log: Callable[[dict], None] | None = None,
        body=None,
    ):
        """footprint_below(top) -> interface/workspace.py Footprint: what of the robot is lower than an obstacle's top
        (inf: all of it), seen from above in the base frame (default: a circle of the config's footprint_radius).
        stop_model (robot/nav/speed.py StopModel, the base's measured stop) and base_accel: the speed limit by the free
        space ahead and the carrot sized for it (none: full speed, the config's lookahead). centre_limit: cells the
        base's centre may not enter (a fence). tuck_arm=False: the base drives with the arm as it is (its footprint
        covers it; the arm moving stops the drive) instead of first moving it to its driving posture. drive_log: called
        with a dict per control tick while the base drives (and as it comes to rest). body: the robot's own parts the
        arm keeps clear of (interface/workspace.py Body, an embodiment's robot.yaml `body:`); its planned arm moves into
        them are refused. None: no body (RoboCasa's PandaOmron)."""
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
        arm = ArmController(env, frames.ik, self._cfg, *_constraints_of(body))
        self._arm = ArmStack(
            env, self._cfg, frames.ik, arm, Gripper(env, self._cfg), frames, ArmCollisionModel(frames.ik.model),
        )  # fmt: skip
        self._grasp_proposer = grasp_proposer
        self._placement_proposer = placement_proposer
        self._fixture_actuator = fixture_actuator
        self._held_object: HeldObject | None = None
        self._loc = Localizer()  # odometry tied to the world by the env's localization (mocap), if it has one
        if footprint_below is None:
            radius = self._cfg.footprint_radius
            footprint_below = lambda top=math.inf: Footprint((0.0, 0.0, 0.0, 0.0), np.array([[0.0, 0.0, radius]]))  # noqa: E731
        self._footprint_below = footprint_below
        self._stop_model, self._base_accel = stop_model, float(base_accel)
        self._centre_limit = centre_limit
        self._tuck_arm = tuck_arm
        self.drive_log = drive_log
        self._space = self._build_space()
        self._motion = MotionManager(self.get_base_pose)

    # -- RobotAPI ---------------------------------------------------------

    def go_to_pose(self, goal: Pose2D, approach_m: float = 0.0) -> None:
        """approach_m > 0: a square finish, the last approach_m straight in along the goal's heading (a station facing
        a counter), after a stop at its start."""
        self._motion.start(lambda preempt: self._run_go_to_pose(goal, preempt, approach_m))

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
        # Burners and appliances through the fixture actuator; utensils (lower, stir, lift); pouring.
        # Not (yet): squeezed / powered gadgets.
        return frozenset({BASE, GRASP, ARTICULATE, HANDHELD_TOOLS, POUR} | ({APPLIANCES} if self._fixture_actuator is not None else set()))

    def use_tool(self, action: str, target: Detection | None = None, seconds: float | None = None) -> None:
        self._motion.start(lambda preempt: self._run_tool(action, target, seconds, preempt))

    def _run_tool(self, action: str, target: Detection | None, seconds: float | None, preempt) -> MotionResult:
        if action == LOWER_INTO and target is not None:
            result = run_lower_into(self._arm, self._placement_proposer, target, self._held, preempt)
        elif action == STIR:
            result = run_stir(self._arm, float(seconds or 20.0), self._held, preempt)
        elif action == LIFT_OUT:
            result = run_lift_out(self._arm, preempt)
        elif action == POUR_INTO and target is not None:
            result = run_pour(self._arm, target, self._held, preempt)
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
        """Where the base is in the world: its odometry, tied to the world by the env's localization (if it has one)."""
        state = self._env.get_state()
        self._loc.update(state)
        return self._loc.world(state.base_pose)

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

    def _run_go_to_pose(self, goal: Pose2D, preempt: threading.Event, approach_m: float = 0.0) -> MotionResult:
        cfg = self._cfg
        plan = self._space.with_margin(cfg.safety_margin)
        if not plan.is_free(goal.x, goal.y, goal.yaw):
            grid = plan.at(goal.yaw)
            cell = grid.nearest_free_cell(goal.x, goal.y, cfg.snap_radius) if cfg.snap_radius > 0 else None
            if cell is None:
                return self._result(
                    MotionStatus.NO_PATH,
                    f"goal is in collision and no free space within {cfg.snap_radius:.2f} m",
                )
            goal = Pose2D(*grid.cell_to_world(cell), goal.yaw)
        for attempt in range(cfg.transient_retries + 1):
            last = attempt == cfg.transient_retries
            try:
                result = self._plan_and_drive(goal, preempt, approach_m)
            except RuntimeError as exc:  # the env failed a tick (the real robot: its link stalled, its server held it)
                if last:
                    raise
                why = f"{type(exc).__name__}: {exc}"
            else:
                if result.status != MotionStatus.NOT_LOCALIZED or last:
                    return result
                why = result.message
            if not self._wait_until_live(preempt, cfg.transient_wait_s):
                return self._result(MotionStatus.STOPPED if preempt.is_set() else MotionStatus.NOT_LOCALIZED,
                                    f"{why}; not back within {cfg.transient_wait_s:.0f} s")  # fmt: skip
        raise AssertionError("unreachable")

    def _wait_until_live(self, preempt: threading.Event, timeout_s: float) -> bool:
        """After a passing stop: wait for the env to answer and the localizer to be live (then a second more)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s and not preempt.is_set():
            try:
                if self._loc.why_not_live(self._env.get_state()) is None:
                    time.sleep(1.0)
                    return not preempt.is_set()
            except RuntimeError:
                pass
            time.sleep(0.1)
        return False

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

    def _plan_and_drive(self, goal: Pose2D, preempt: threading.Event, approach_m: float = 0.0) -> MotionResult:
        """Plan a path clear for the footprint at the heading the base will have along it (robot/nav/heading_plan.py),
        keeping the config's safety margin, then drive it (and the straight approach, if any)."""
        lost = self._loc.why_not_live(self._env.get_state())
        if lost is not None:
            return self._result(MotionStatus.NOT_LOCALIZED, lost)
        pose = self.get_base_pose()
        body, plan = self._space, self._space.with_margin(self._cfg.safety_margin)
        if not body.is_free(pose.x, pose.y, pose.yaw):
            return self._result(MotionStatus.NO_PATH, "base is in collision on the map")
        pre = Pose2D(goal.x - approach_m * math.cos(goal.yaw), goal.y - approach_m * math.sin(goal.yaw), goal.yaw)
        if approach_m > 0 and not body.at(goal.yaw).segment_free((pre.x, pre.y), (goal.x, goal.y)):
            return self._result(MotionStatus.NO_PATH, "the straight approach to the goal is not clear")
        begin = escape_start(plan.at(pose.yaw), body.at(pose.yaw), (pose.x, pose.y))
        if begin is None:
            return self._result(MotionStatus.NO_PATH, "base is inside the margin with no clear way out")
        points, why = None, None
        last = getattr(self, "_last_plan", None)
        if last is not None and approach_m == 0:  # choose_stance just planned this: reuse it if the base hasn't moved
            (sx, sy, syaw), (gx, gy, gyaw), lbegin, lpoints = last
            if math.hypot(pose.x - sx, pose.y - sy) < 0.02 and abs(wrap_angle(pose.yaw - syaw)) < math.radians(1) \
                    and math.hypot(goal.x - gx, goal.y - gy) < 1e-6 and abs(wrap_angle(goal.yaw - gyaw)) < 1e-6:
                points = lpoints
        self._last_plan = None
        if points is None:
            points, why = plan_at_headings(plan, body, pose, begin, pre, goal.yaw, style=self._style())
        if points is None:
            blocking = self._doors_in_the_way(lambda planner: _has_path(planner, pose, goal))
            return self._result(MotionStatus.NO_PATH, why + blocking)
        path = headings(pose, subdivide(points, 1.0), goal.yaw, self._style(), body)
        if self._tuck_arm:
            # Interlock: the arm is tucked (carrying, if it holds something) before the base moves.
            tucked = tuck_arm_for_driving(self._arm.arm, self._cfg, self._held, preempt)
            if not tucked.ok:
                return self._from_arm(tucked)
        path_m = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(path, path[1:]))
        result = self._drive(path, preempt) if len(path) >= 2 else self._result(MotionStatus.SUCCESS, "already there")
        if result.status == MotionStatus.SUCCESS and approach_m > 0:
            result = self._drive([pre, goal], preempt, leg=1)
        result.data.update(path=[(p.x, p.y) for p in path] + ([(goal.x, goal.y)] if approach_m > 0 else []),
                           path_m=path_m + approach_m)  # fmt: skip
        return result

    def guard(self) -> BaseGuard:
        """The base's stops and speed limit for a drive (robot/safety.py), on this robot's map and footprint."""
        return BaseGuard(self._space, self._loc, self._stop_model, self._base_accel,
                         arm_moved_rad=None if self._tuck_arm else _ARM_MOVED_RAD, centre_limit=self._centre_limit,
                         plan_space=self._space.with_margin(self._cfg.safety_margin))  # fmt: skip

    def _drive(self, path: list[Pose2D], preempt: threading.Event, leg: int = 0) -> MotionResult:
        env, cfg = self._env, self._cfg
        controller = BaseController(path, cfg)
        guard = self.guard()
        dt = cfg.control_dt
        goal = path[-1]
        straight = len(path) == 2
        turn = abs(wrap_angle(goal.yaw - path[0].yaw))
        expected = max(controller.length / max(cfg.max_lin_vel, 1e-6), turn / max(cfg.max_ang_vel, 1e-6))
        timeout = min(cfg.motion_timeout_s, 10.0 + 3.0 * expected)
        start_time: float | None = None
        settled_since: float | None = None
        best = math.inf
        guard.begin(env.get_state())
        hold = None  # cfg.arrive_hold: the odometry pose held once inside the tolerances
        try:
            while True:
                if preempt.is_set():
                    status, message = MotionStatus.STOPPED, "motion was stopped"
                    break
                state = env.get_state()
                self._loc.update(state)
                pose = self._loc.world(state.base_pose)
                if start_time is None:
                    start_time = state.time
                elapsed = state.time - start_time
                stop = guard.stop_reason(state, pose)
                if stop is not None:
                    status = MotionStatus.NOT_LOCALIZED if "localization" in stop else MotionStatus.BLOCKED
                    message = stop
                    break
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
                speed = math.hypot(*state.base_velocity[:2])
                still = cfg.arrive_still_mps is None or speed < cfg.arrive_still_mps
                inside = controller.at_goal(pose)
                if cfg.arrive_hold and (inside or hold is not None):
                    if hold is None:  # first tick inside the tolerances: the goal frozen in odometry, never re-tied
                        hold = self._loc.to_odom(goal)
                    if still:
                        settled_since = elapsed if settled_since is None else settled_since
                        if elapsed - settled_since >= cfg.arrive_settle_s:
                            status, message = MotionStatus.SUCCESS, "reached the goal (held inside the tolerances)"
                            break
                    else:
                        settled_since = None
                    if elapsed > timeout:
                        status, message = MotionStatus.TIMEOUT, f"motion timed out after {timeout:.0f} s"
                        break
                    self._log(leg, "hold", elapsed, state, pose, goal)
                    env.set_base_target(hold, (0.05, 0.05, cfg.max_ang_vel))
                    env.wait_for_next_tick(dt)
                    continue
                if controller.at_goal(pose) and still:
                    settled_since = elapsed if settled_since is None else settled_since
                    if elapsed - settled_since >= cfg.arrive_settle_s:
                        status, message = MotionStatus.SUCCESS, "reached the goal"
                        break
                else:
                    settled_since = None
                if elapsed > timeout:
                    status, message = MotionStatus.TIMEOUT, f"motion timed out after {timeout:.0f} s"
                    break
                dist = pose.distance_to(goal)
                best = min(best, dist)
                if straight and dist > best + cfg.moving_away_m:
                    status, message = MotionStatus.BLOCKED, f"moving away from the goal ({dist:.2f} m, best {best:.2f} m)"
                    break
                # A base that drives to a pose itself gets the carrot; any other, a velocity.
                target, look = guard.carrot(controller, pose, cfg.max_lin_vel, cfg.lookahead)
                if controller.deviation > cfg.deviation_threshold:
                    status = MotionStatus.BLOCKED
                    message = f"base is {controller.deviation:.2f} m off the planned path"
                    break
                bv = state.base_velocity  # in odometry: the way it is moving, turned into the world
                c, s = math.cos(self._loc.world_from_odom.yaw), math.sin(self._loc.world_from_odom.yaw)
                way = (c * bv[0] - s * bv[1], s * bv[0] + c * bv[1]) if speed > 0.02 else (target.x - pose.x, target.y - pose.y)
                v_cap, free = guard.speed_cap(pose, way, cfg.max_lin_vel)
                self._log(leg, "drive", elapsed, state, pose, goal, controller.deviation, target, look, v_cap, free)
                creep = max(v_cap, 0.02)  # never 0: a base that plans to rest at its target still creeps on
                if not env.set_base_target(self._loc.to_odom(target), (creep, creep, cfg.max_ang_vel)):
                    vx, vy, vyaw = controller.step(pose, dt, look)
                    n = math.hypot(vx, vy)
                    if n > v_cap:
                        vx, vy = vx * v_cap / n, vy * v_cap / n
                    env.set_base_velocity(*self._loc.velocity_to_odom(vx, vy), vyaw, frame="world")
                env.wait_for_next_tick(dt)
        finally:
            # Velocity is zeroed on every exit, including exceptions.
            env.set_base_velocity(0.0, 0.0, 0.0)
        self._settle(dt, leg, goal)
        return self._result(status, message)

    def _log(self, leg, phase, t, state, pose, goal, deviation=None, target=None, look=None, v_cap=None, free=None):
        if self.drive_log is None:
            return
        r = state.localization
        row = {"leg": leg, "t": t, "phase": phase, "pose": [pose.x, pose.y, pose.yaw],
               "odom": [state.base_pose.x, state.base_pose.y, state.base_pose.yaw],
               "base_velocity": [float(v) for v in state.base_velocity],
               "dist": pose.distance_to(goal), "yaw_err_deg": math.degrees(wrap_angle(goal.yaw - pose.yaw)),
               "localization_age_s": None if r is None else r.age_s, "off_line": deviation}  # fmt: skip
        if target is not None:
            row.update(carrot=[target.x, target.y, target.yaw], lookahead=look, v_cap=v_cap, free_m=free)
        self.drive_log(row)

    def _settle(self, dt: float, leg: int = 0, goal: Pose2D | None = None) -> None:
        """Tick with a zero command until the base has come to rest, so the
        reported pose is where the robot actually ends up."""
        for _ in range(_MAX_SETTLE_TICKS):
            self._env.wait_for_next_tick(dt)
            if self.drive_log is not None and goal is not None:
                try:
                    state = self._env.get_state()
                    self._log(leg, "hold", state.time, state, self._loc.world(state.base_pose), goal)
                except Exception:  # the env may be gone (a stop test): the log is best-effort
                    pass
            if base_at_rest(self._env):
                break

    def choose_stance(self, candidates: list[Pose2D]) -> tuple[Pose2D | None, str]:
        """Of base poses (each at its own heading), the one clear for the footprint with the margin and cheapest to
        reach by A* from where the base is: (pose, "") or (None, why not)."""
        plan, body = self._space.with_margin(self._cfg.safety_margin), self._space
        clear = [c for c in candidates if plan.is_free(c.x, c.y, c.yaw)]
        if not clear:
            return None, f"no stance clear for the footprint ({len(candidates)} candidates, all blocked)"
        pose = self.get_base_pose()
        begin = escape_start(plan.at(pose.yaw), body.at(pose.yaw), (pose.x, pose.y))
        if begin is None:
            return None, "the base is inside the margin with no clear way out"
        yaws = [y for c in clear for y in plan.sweep(pose.yaw, c.yaw)]
        planner = AStarPlanner(plan.blocked_for_all(yaws))
        costs = planner.cost_to(begin, {planner.grid.world_to_cell(c.x, c.y) for c in clear})
        ranked = sorted((costs[cell], i) for i, c in enumerate(clear) if (cell := planner.grid.world_to_cell(c.x, c.y)) in costs)
        for _, i in ranked:
            c = clear[i]
            found = plan_at_headings(plan, body, pose, begin, c, c.yaw, style=self._style())[0]
            if found is not None:
                self._last_plan = ((pose.x, pose.y, pose.yaw), (c.x, c.y, c.yaw), begin, found)
                return c, ""
        return None, "no stance reachable from here"

    def _build_space(self) -> FootprintSpace:
        grid = self._grid_with(list(self._extra_obstacles.values()))
        space = FootprintSpace.by_height(grid, self._footprint_below)
        space.heading_zones = list(getattr(self._grid, "heading_zones", [])) + list(getattr(self, "heading_zones", []))  # the map's, then any set
        return space

    def _style(self) -> PathStyle | None:
        c = self._cfg
        if c.path_clearance_m <= 0 and not c.turn_in_room:
            return None
        return PathStyle(clearance_m=c.path_clearance_m, turn_in_room=c.turn_in_room, need_m=0.369 + c.safety_margin)

    def set_heading_zones(self, zones) -> None:
        """More places where the base may only face some ways than the map's own (OccupancyGrid.heading_zones):
        [(cells grid, [(yaw rad, tolerance rad), ...])] (FootprintSpace)."""
        self.heading_zones = list(zones)
        self._space = self._build_space()

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
        heights = self._grid.heights.copy()
        for points in extra:
            for x, y in points:
                cell = self._grid.world_to_cell(float(x), float(y))
                if self._grid.in_bounds(cell):
                    occupied[cell] = True
                    heights[cell] = np.inf  # an open door: never reached over
        return OccupancyGrid(occupied, self._grid.resolution, self._grid.origin, heights)

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
        self._space = self._build_space()

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


def _constraints_of(body) -> tuple[list, float]:
    """The arm's constraints and hand length for a body (none without one)."""
    from chefbot.embodiments.constraints import for_robot

    return for_robot(body), (body.hand_m if body is not None else 0.0)
