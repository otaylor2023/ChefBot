"""RobotAPI: the only thing the agent knows about the robot."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chefbot.interface.types import Detection, HeldObject, MotionResult, Pose2D

# What a robot can do (RobotAPI.capabilities). The agent offers a tool only when
# the robot has every capability the tool needs.
BASE = "base"  # drive: go_to_pose, align_to_point
GRASP = "grasp"  # pick, place
ARTICULATE = "articulate"  # open / close doors and drawers
APPLIANCES = "appliances"  # set_power: burners, toaster, ...
HANDHELD_TOOLS = "handheld_tools"  # use_tool: lower_into, lift_out, stir (a spoon, a spatula)
TRIGGER_TOOLS = "trigger_tools"  # use_tool: squeeze, release (powered or squeezed gadgets)
POUR = "pour"  # use_tool: pour
ALL_CAPABILITIES = frozenset({BASE, GRASP, ARTICULATE, APPLIANCES, HANDHELD_TOOLS, TRIGGER_TOOLS, POUR})

# use_tool actions.
LOWER_INTO, LIFT_OUT, SQUEEZE, RELEASE, POUR_INTO, STIR = "lower_into", "lift_out", "squeeze", "release", "pour", "stir"


class RobotAPI(ABC):
    """Every motion call is non-blocking and shares one motion thread: a new
    call preempts the motion in progress, and `wait` / `get_status` report on
    whichever was started last."""

    def capabilities(self) -> frozenset[str]:
        """What this robot can do (the constants above). Tools for anything else
        are not offered to the agent."""
        return frozenset({BASE, GRASP, ARTICULATE, APPLIANCES})

    @abstractmethod
    def go_to_pose(self, goal: Pose2D) -> None:
        """Non-blocking. The robot plans a path to `goal` and drives it."""

    @abstractmethod
    def align_to_point(self, target: np.ndarray) -> None:
        """Non-blocking. The robot picks a reachable standoff pose around
        `target` (world xyz), plans to it and drives it."""

    @abstractmethod
    def wait(self, timeout_s: float | None = None) -> MotionResult:
        """Block until the current motion ends (None) or `timeout_s` elapses.
        An expired timeout returns status "moving"; the motion continues."""

    @abstractmethod
    def get_status(self) -> MotionResult:
        """Non-blocking snapshot of the current / last motion."""

    @abstractmethod
    def stop(self) -> None:
        """Preempt any motion; halts the base and the arm. Callable at any time,
        from any thread."""

    @abstractmethod
    def pick(self, target: Detection) -> None:
        """Non-blocking. Grasp the object at `target` and lift it. Does not
        move the base."""

    @abstractmethod
    def place(self, surface: Detection) -> None:
        """Non-blocking. Set the held object down on `surface`. Does not move the base."""

    @abstractmethod
    def actuate_fixture(self, fixture: Detection, open: bool) -> None:
        """Non-blocking. Open or close `fixture` (the fixture or its handle)."""

    @abstractmethod
    def set_power(self, appliance: Detection, on: bool) -> None:
        """Non-blocking. Turn a burner (or another appliance: toaster, microwave)
        on or off. Errors as for actuate_fixture."""

    def use_tool(self, action: str, target: Detection | None = None, seconds: float | None = None) -> None:
        """Non-blocking. Work the held tool or container: LOWER_INTO / POUR_INTO a
        target, LIFT_OUT, SQUEEZE / RELEASE its trigger or handles (not the tool),
        STIR for `seconds` with a spoon lowered into something. Robots without
        HANDHELD_TOOLS / POUR end with status "unsupported"."""
        raise NotImplementedError(f"{type(self).__name__} cannot use handheld tools")

    @abstractmethod
    def get_held(self) -> HeldObject | None:
        """What the gripper is holding, if anything."""

    @abstractmethod
    def get_base_pose(self) -> Pose2D:
        """World frame; x, y in meters, yaw in radians."""
