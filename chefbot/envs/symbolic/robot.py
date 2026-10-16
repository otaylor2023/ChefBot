"""SymbolicRobot: RobotAPI on the symbolic kitchen, and its clock.

Motions take run time (driving at ~0.5 m/s, a pick ~10 s) and change the world
when they end. `wait()` lets that time pass, so cooking and timers progress and
events can arrive mid-drive. Failures come from the world (closed doors, an
occupied burner, out of reach) and from injected rates (task file).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from chefbot.envs.symbolic.world import CRACKER, OPENABLE, POWERED, UNPICKABLE, UTENSILS, Rates, World, label_of
from chefbot.interface.clock import Clock
from chefbot.interface.robot_api import (
    ALL_CAPABILITIES,
    LIFT_OUT,
    LOWER_INTO,
    POUR_INTO,
    RELEASE,
    SQUEEZE,
    STIR,
    RobotAPI,
)
from chefbot.interface.types import Detection, HeldObject, MotionResult, MotionStatus, Pose2D

ALIGN_STANDOFF = 0.6  # m in front of the target after align
REACH = 0.85  # m: farthest target (xy, from the base center) the arm can work on
REACH_ANGLE = math.radians(60)  # off the base's heading
MATCH = 0.3  # m: a detection names the nearest object of its label within this


class SymbolicClock(Clock):
    def __init__(self, world: World):
        self._world = world

    def now(self) -> float:
        return self._world.t

    def sleep(self, seconds: float) -> None:
        self._world.advance(max(0.0, float(seconds)))


@dataclass
class _Motion:
    start: Pose2D
    goal: Pose2D
    t0: float
    t1: float
    effect: Callable[[], MotionResult | None] | None  # runs at t1; a result overrides success


class SymbolicRobot(RobotAPI):
    def __init__(self, world: World, pose: Pose2D, rates: Rates | None = None, rng: np.random.Generator | None = None):
        self.world = world
        self._pose = pose
        self.rates = rates or Rates()
        self._rng = rng or np.random.default_rng(0)
        self._motion: _Motion | None = None
        self._result = MotionResult(MotionStatus.IDLE, pose, "")
        self.picked_from: np.ndarray | None = None

    # -- RobotAPI: motions ------------------------------------------------------

    def go_to_pose(self, goal: Pose2D) -> None:
        if self._roll(self.rates.nav):
            self._fail_after(2.0, MotionStatus.BLOCKED, "something is in the way")
            return
        self._drive(goal)

    def align_to_point(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        fixture = min(self.world.fixtures.values(), key=lambda f: float(np.linalg.norm(f.location - target[:2])))
        xy = target[:2] + ALIGN_STANDOFF * fixture.facing
        self._drive(Pose2D(float(xy[0]), float(xy[1]), math.atan2(-fixture.facing[1], -fixture.facing[0])),
                    extra=self._t("drive_overhead") + self._t("align"))  # fmt: skip

    def pick(self, target: Detection) -> None:
        def effect() -> MotionResult | None:
            world = self.world
            if world.held is not None:
                return self._r(MotionStatus.ALREADY_HOLDING, f"already holding the {world.things[world.held].kind}")
            if target.label in UNPICKABLE:
                return self._r(MotionStatus.NO_GRASP, f"the {target.label} is not something the gripper can pick up")
            name = self._find(target)
            if name is None:
                return self._r(MotionStatus.NO_GRASP, f"no {target.label} where it was seen")
            if not self._in_reach(world.position(name)):
                return self._r(MotionStatus.UNREACHABLE, f"the {target.label} is out of reach; align to it first")
            if world.enclosed(name):
                return self._r(MotionStatus.UNREACHABLE, f"the {target.label} is behind a closed door")
            if self._roll(self.rates.grasp):
                return self._r(MotionStatus.GRASP_FAILED, "the gripper closed on nothing")
            self.picked_from = world.position(name)
            world.move(name, None)
            return None

        self._act(self._t("pick"), effect, "holding it")

    def place(self, surface: Detection) -> None:
        def effect() -> MotionResult | None:
            world = self.world
            if world.held is None:
                return self._r(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
            place = self._find_place(surface)
            if place is None:
                return self._r(MotionStatus.NO_PLACEMENT, f"no {surface.label} where it was seen")
            if not self._in_reach(world.position(place)):
                return self._r(MotionStatus.UNREACHABLE, f"the {surface.label} is out of reach; align to it first")
            why = world.can_place(world.held, place)
            if why is not None:
                return self._r(MotionStatus.NO_PLACEMENT, why)
            held = world.held
            if world.things[held].kind == "egg" and world.kind_of(place) == CRACKER and self._roll(self.rates.crack_early):
                world.remove(held)
                world.events.append("an egg cracked too early, going into the egg cracker, and spilled")
                return self._r(MotionStatus.FAILED, "the egg cracked as it went into the egg cracker and spilled: it is gone")
            if self._roll(self.rates.place):
                fallback = world.fixture_of(place)
                if fallback is not None and fallback.name != place and world.can_place(held, fallback.name) is None:
                    world.move(held, fallback.name, self._slot_near(fallback.name))
                    return None  # reported as a success: only ground truth knows
            world.move(held, place, self._slot_near(place))
            return None

        self._act(self._t("place"), effect, "put it down")

    def actuate_fixture(self, fixture: Detection, open: bool) -> None:
        def effect() -> MotionResult | None:
            name = self._find_fixture(fixture, OPENABLE)
            if name is None:
                return self._r(MotionStatus.UNSUPPORTED, f"the {fixture.label} is not something that opens")
            target = self.world.fixtures[name]
            if not self._in_reach(fixture.position_world):  # where it was seen: its handle, or its middle
                return self._r(MotionStatus.UNREACHABLE, f"the {target.kind} is out of reach; align to it first")
            state = "open" if open else "closed"
            if self._roll(self.rates.actuate):
                return self._r(MotionStatus.FAILED, "the gripper slipped off the handle; try again")
            if target.open == bool(open):
                return self._r(MotionStatus.SUCCESS, f"the {target.kind} is already {state}", scripted=True)
            target.open = bool(open)
            return self._r(MotionStatus.SUCCESS, f"the {target.kind} is now {state} (scripted)", scripted=True)

        self._act(self._t("open_close"), effect, "")

    def set_power(self, appliance: Detection, on: bool) -> None:
        def effect() -> MotionResult | None:
            world = self.world
            if appliance.label == "burner":
                name = self._nearest(appliance.position_world, list(world.burners), MATCH)
            else:
                name = self._find_fixture(appliance, POWERED)
            if name is None:
                return self._r(MotionStatus.UNSUPPORTED, f"the {appliance.label} is not something that turns on")
            if not self._in_reach(world.position(name)):
                return self._r(MotionStatus.UNREACHABLE, f"the {appliance.label} is out of reach; align to it first")
            state = "on" if on else "off"
            if self._roll(self.rates.power):
                return self._r(MotionStatus.FAILED, f"the {appliance.label} did not take; try again")
            is_on = world.burners[name].on if name in world.burners else bool(world.fixtures[name].powered)
            if is_on == bool(on):
                return self._r(MotionStatus.SUCCESS, f"the {appliance.label} is already {state}", scripted=True)
            world.set_power(name, on)
            return self._r(MotionStatus.SUCCESS, f"the {appliance.label} is now {state} (scripted)", scripted=True)

        self._act(self._t("power"), effect, "")

    def capabilities(self) -> frozenset[str]:
        return ALL_CAPABILITIES

    def use_tool(self, action: str, target: Detection | None = None, seconds: float | None = None) -> None:
        if action == STIR:
            self._stir(float(seconds or 10.0))
            return

        def effect() -> MotionResult | None:
            world = self.world
            if world.held is None:
                return self._r(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
            place = None
            if action in (LOWER_INTO, POUR_INTO):
                place = self._find_place(target) if target is not None else None
                if place is None:
                    return self._r(MotionStatus.NO_PLACEMENT, f"no {getattr(target, 'label', 'target')} where it was seen")
                if not self._in_reach(world.position(place)):
                    return self._r(MotionStatus.UNREACHABLE, f"the {target.label} is out of reach; align to it first")
            if action in (LOWER_INTO, SQUEEZE, POUR_INTO) and self._roll(self.rates.tool):
                kind = world.things[world.held].kind
                return self._r(MotionStatus.FAILED, f"the {kind} slipped in the gripper; try again")
            if action == LOWER_INTO:
                why = world.lower_into(place)
                if why is not None:
                    return self._r(MotionStatus.NO_PLACEMENT, why)
                return self._r(MotionStatus.SUCCESS, f"lowered into the {target.label}")
            if action == LIFT_OUT:
                splat = world.lift_out()
                return self._r(MotionStatus.SUCCESS, splat or "lifted out")
            if action == SQUEEZE:
                return self._r(MotionStatus.SUCCESS, world.squeeze())
            if action == RELEASE:
                world.release()
                return self._r(MotionStatus.SUCCESS, "released")
            if action == POUR_INTO:
                why = world.pour(place)
                if why is not None:
                    return self._r(MotionStatus.NO_PLACEMENT, why)
                return self._r(MotionStatus.SUCCESS, f"poured into the {target.label}")
            return self._r(MotionStatus.UNSUPPORTED, f"unknown tool action {action}")

        self._act(self._t(action, 3.0), effect, action)

    def _stir(self, seconds: float) -> None:
        """Stir for `seconds`: the world stirs while the motion runs, and stops when it ends or is stopped."""
        world = self.world
        problem = None
        if world.held is None:
            problem = self._r(MotionStatus.NOT_HOLDING, "the gripper is not holding anything")
        elif world.things[world.held].kind not in UTENSILS:
            problem = self._r(MotionStatus.UNSUPPORTED, f"cannot stir with a {world.things[world.held].kind}; use a spoon or spatula")
        elif world.tool_into is None:
            problem = self._r(MotionStatus.NO_PLACEMENT, "the spoon is not in anything; lower_into a bowl or pan first")
        if problem is None and self._roll(self.rates.tool):
            problem = self._r(MotionStatus.FAILED, "the spoon slipped in the gripper; try again")
        if problem is not None:
            self._act(1.0, lambda: problem, STIR)
            return

        def done() -> MotionResult:
            report = world.stir_report()
            world.stirring = False
            return self._r(MotionStatus.SUCCESS, f"{report} for {seconds:.0f} s")

        self._act(seconds, done, STIR)
        world.stirring = True

    # -- RobotAPI: status ---------------------------------------------------------

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        motion = self._motion
        if motion is None:
            return self._result
        clock = self.world
        until = motion.t1 if timeout_s is None else min(motion.t1, clock.t + float(timeout_s))
        clock.advance(max(0.0, until - clock.t))
        self._settle()
        return self.get_status()

    def get_status(self) -> MotionResult:
        self._settle()
        if self._motion is not None:
            return MotionResult(MotionStatus.MOVING, self.get_base_pose(), "")
        return self._result

    def stop(self) -> None:
        self._settle()
        if self._motion is not None:
            self._pose = self.get_base_pose()
            self._motion = None
            self.world.stirring = False
            self._result = MotionResult(MotionStatus.STOPPED, self._pose, "stopped")

    def get_held(self) -> HeldObject | None:
        held = self.world.held
        if held is None:
            return None
        return HeldObject(self.world.things[held].kind, 0.05, self.picked_from if self.picked_from is not None else np.zeros(3))

    def get_base_pose(self) -> Pose2D:
        motion = self._motion
        if motion is None:
            return self._pose
        f = min(1.0, max(0.0, (self.world.t - motion.t0) / max(motion.t1 - motion.t0, 1e-9)))
        dyaw = math.atan2(math.sin(motion.goal.yaw - motion.start.yaw), math.cos(motion.goal.yaw - motion.start.yaw))
        return Pose2D(
            motion.start.x + f * (motion.goal.x - motion.start.x),
            motion.start.y + f * (motion.goal.y - motion.start.y),
            motion.start.yaw + f * dyaw,
        )

    # -- internals --------------------------------------------------------------------

    def _settle(self) -> None:
        """Finish a motion whose time is up."""
        motion = self._motion
        if motion is None or self.world.t < motion.t1 - 1e-9:
            return
        self._motion = None
        self._pose = motion.goal
        result = motion.effect() if motion.effect is not None else None
        self._result = result or MotionResult(MotionStatus.SUCCESS, self._pose, "")
        if self._result.final_pose is None:
            self._result.final_pose = self._pose

    def _start(self, goal: Pose2D, duration: float, effect=None) -> None:
        self._settle()
        start = self.get_base_pose()
        if self._motion is not None:  # a new command preempts the old one
            self._pose = start
            self.world.stirring = False
        self._motion = _Motion(start, goal, self.world.t, self.world.t + duration, effect)

    def _t(self, key: str, default: float = 0.0) -> float:
        """A duration or speed from the world's timing (timing.yaml)."""
        return float(self.world.timing.robot.get(key, default))

    def _drive(self, goal: Pose2D, extra: float | None = None) -> None:
        self.world.lift_out()  # a lowered tool comes out before the base moves
        start = self.get_base_pose()
        distance = math.hypot(goal.x - start.x, goal.y - start.y)
        turn = abs(math.atan2(math.sin(goal.yaw - start.yaw), math.cos(goal.yaw - start.yaw)))
        extra = self._t("drive_overhead") if extra is None else extra
        self._start(goal, distance / self._t("drive_speed", 0.35) + turn / self._t("turn_speed", 0.8) + extra)

    def _act(self, duration: float, effect, _what: str) -> None:
        here = self.get_base_pose()
        self._start(here, duration, effect)

    def _fail_after(self, duration: float, status: str, message: str) -> None:
        here = self.get_base_pose()
        self._start(here, duration, lambda: self._r(status, message))

    def _r(self, status: str, message: str, **data) -> MotionResult:
        return MotionResult(status, self._pose, message, dict(data))

    def _roll(self, p: float) -> bool:
        return p > 0 and float(self._rng.random()) < p

    def _in_reach(self, position: np.ndarray) -> bool:
        pose = self._pose
        dx, dy = float(position[0]) - pose.x, float(position[1]) - pose.y
        if math.hypot(dx, dy) > REACH:
            return False
        off = math.atan2(dy, dx) - pose.yaw
        return abs(math.atan2(math.sin(off), math.cos(off))) <= REACH_ANGLE

    def _nearest(self, position, names: list[str], radius: float) -> str | None:
        position = np.asarray(position, dtype=float)
        best = min(names, key=lambda n: float(np.linalg.norm(self.world.position(n)[:2] - position[:2])), default=None)
        if best is None or float(np.linalg.norm(self.world.position(best)[:2] - position[:2])) > radius:
            return None
        return best

    def _find(self, det: Detection) -> str | None:
        """The loose object a detection names."""
        names = [t.name for t in self.world.things.values() if t.kind == det.label and t.where is not None]
        return self._nearest(det.position_world, names, MATCH)

    def _find_fixture(self, det: Detection, kinds: set[str]) -> str | None:
        label = det.label.removesuffix(" handle")
        names = [f.name for f in self.world.fixtures.values() if f.kind == label and f.kind in kinds]
        return self._nearest(det.position_world, names, 1.0)

    def _find_place(self, det: Detection) -> str | None:
        if det.label == "burner":
            return self._nearest(det.position_world, list(self.world.burners), MATCH)
        fixtures = [f.name for f in self.world.fixtures.values() if f.kind == label_of(det.label) or f.kind == det.label]
        if fixtures:
            return self._nearest(det.position_world, fixtures, 1.0)
        things = [t.name for t in self.world.things.values() if t.kind == det.label and t.where is not None]
        return self._nearest(det.position_world, things, MATCH)

    def _slot_near(self, place: str) -> float | None:
        fixture = self.world.fixtures.get(place)
        if fixture is None:
            return None
        return float((np.array([self._pose.x, self._pose.y]) - fixture.location) @ fixture.across)
