"""The real robot as an agent's world when what the agent calls are chefbot-v1 skills (run_agent --env real): the
skills move the robot, each as one sandboxed episode on the real host (runtime/agent_skills.py), and the agent's
loop gets what it asks of a world from here. A RobotAPI that does nothing by itself (no RobotAPI tool is offered in
a chefbot-v1 session: its capabilities are none), no detector, no map, the wall clock, and no ground truth: a step
the agent completes is taken as done, unchecked (NoGroundTruth; agent/check UncheckedJudge).
"""

from __future__ import annotations

import time

from chefbot.interface.clock import Clock
from chefbot.interface.detector import Detector
from chefbot.interface.observer import Observation, Observer
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import Detection, HeldObject, MotionResult, MotionStatus, Pose2D


class StationaryRobot(RobotAPI):
    """The real host's robot, for the loop's own questions (where is the base, is anything moving) and its stop."""

    def __init__(self, host):
        self._host = host

    def capabilities(self) -> frozenset[str]:
        return frozenset()

    def get_base_pose(self) -> Pose2D:
        return self._host.env.get_state().base_pose

    def get_status(self) -> MotionResult:
        return MotionResult(MotionStatus.IDLE, message="the skills move the robot; between them it holds")

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        return self.get_status()

    def stop(self) -> None:
        self._host.halt()  # hold the arm where it is

    def get_held(self) -> HeldObject | None:
        return None  # the agent hears what is held from the skills' results

    def _none(self, *args, **kwargs) -> None:
        raise RuntimeError("a chefbot-v1 session's robot moves only through its skills")

    go_to_pose = align_to_point = pick = place = actuate_fixture = set_power = _none


class NoDetector(Detector):
    def detect(self, classes: list[str]) -> list[Detection]:
        return []


class WallClock(Clock):
    def __init__(self):
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class NoGroundTruth(Observer):
    """Nothing knows the real kitchen's state: the judge is told so (UncheckedJudge takes the step as done)."""

    def observe(self) -> Observation:
        return Observation(text="(the real robot: no ground truth)")


def no_safety_events(t: float) -> list:
    return []  # the real host's guard stops the arm itself (SafetyStop); nothing for the loop's bus


def unjudged(called: list[str]) -> dict:
    """The run's verdict on the real robot: nothing to judge it against."""
    return {"success": None, "missing": [], "forbidden_present": [], "judged": False}
