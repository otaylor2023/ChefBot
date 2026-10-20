"""Manipulation tools: pick, place, open, close, holding.

Like the nav tools, these only choose what to act on (a visible id); the
robot layer decides how. Nothing here knows a grasp pose or a joint angle.
"""

from __future__ import annotations

from chefbot.agent.tools.nav import NavTools
from chefbot.agent.world.detections import DetectionEntry
from chefbot.agent.world.kitchen import YOU
from chefbot.interface.robot_api import LIFT_OUT, LOWER_INTO, POUR_INTO, RELEASE, SQUEEZE, STIR, RobotAPI
from chefbot.interface.types import Detection, MotionResult, MotionStatus, ToolResult


class ManipTools:
    def __init__(self, robot: RobotAPI, nav: NavTools):
        self._robot = robot
        self._nav = nav  # shares the detections, the map and the held object

    # -- pick / place ---------------------------------------------------------

    def pick(self, id: str, wait: bool = True) -> ToolResult:
        nav = self._nav
        entry = nav.visible_entry(id)
        if isinstance(entry, ToolResult):
            return entry
        if nav.held is not None:
            return ToolResult(
                False,
                MotionStatus.ALREADY_HOLDING,
                f'Already holding {nav.held["id"]}; place it somewhere before picking up something else.',
            )

        def on_done(result: MotionResult) -> None:
            if self._robot.get_held() is None:
                return
            # In the gripper now: it is no longer a thing in the scene to align to or pick.
            nav.held = {"id": entry.map_name or entry.id, "label": entry.label}
            nav.detections.remove(entry.id)
            if entry.map_name is not None:
                nav._map.remove(entry.map_name)
            nav.record_pick(nav.held["id"])

        self._robot.pick(_detection(entry))
        return nav.motion_tool_result(wait, f'pick("{id}")', {"id": id}, on_done)

    def place(self, id: str, wait: bool = True) -> ToolResult:
        nav = self._nav
        entry = nav.visible_entry(id)
        if isinstance(entry, ToolResult):
            return entry
        if nav.held is None:
            return ToolResult(False, MotionStatus.NOT_HOLDING, "The gripper is not holding anything.")
        held = nav.held

        def on_done(result: MotionResult) -> None:
            if self._robot.get_held() is None:
                nav.held = None
                nav.tool_use.update(lowered_into=None, squeezing=False)
                if result.status == MotionStatus.SUCCESS:
                    nav.record_place(held["id"], entry)  # back on the map, where you put it
                # (dropped: it shows up again in the next detect)

        self._robot.place(_detection(entry))
        return nav.motion_tool_result(wait, f'place("{id}")', {"id": id, "object": held["id"]}, on_done)

    def holding(self) -> ToolResult:
        held = self._nav.held
        if held is None:
            return ToolResult(True, None, "Holding nothing.", {"holding": None})
        return ToolResult(True, None, f'Holding {held["id"]}.', {"holding": dict(held)})

    # -- open / close -----------------------------------------------------------

    def open(self, id: str, wait: bool = True) -> ToolResult:
        return self._actuate(id, True, wait)

    def close(self, id: str, wait: bool = True) -> ToolResult:
        return self._actuate(id, False, wait)

    def _actuate(self, id: str, open: bool, wait: bool) -> ToolResult:
        verb = "open" if open else "close"
        entry = self._nav.visible_entry(id)
        if isinstance(entry, ToolResult) and not open and id in self._nav.opened_by:
            entry = self._nav.opened_by[id]  # a door you opened: closed by the handle you opened it with
        if isinstance(entry, ToolResult):
            return entry
        self._robot.actuate_fixture(_detection(entry), open)

        def on_done(result: MotionResult) -> None:
            if result.status == MotionStatus.SUCCESS:
                self._nav.record_state(entry, "open" if open else "closed")

        return self._nav.motion_tool_result(wait, f'{verb}("{id}")', {"id": id}, on_done)


class KitchenTools:
    """turn_on / turn_off: burners and appliances (AGENT_SPEC §7.1)."""

    def __init__(self, robot: RobotAPI, nav: NavTools):
        self._robot = robot
        self._nav = nav

    def turn_on(self, id: str, wait: bool = True) -> ToolResult:
        return self._power(id, True, wait)

    def turn_off(self, id: str, wait: bool = True) -> ToolResult:
        return self._power(id, False, wait)

    def _power(self, id: str, on: bool, wait: bool) -> ToolResult:
        entry = self._nav.visible_entry(id)
        if isinstance(entry, ToolResult):
            return entry
        self._robot.set_power(_detection(entry), on)

        def on_done(result: MotionResult) -> None:
            if result.status == MotionStatus.SUCCESS:
                self._nav.record_state(entry, "on" if on else "off")

        return self._nav.motion_tool_result(wait, f'turn_{"on" if on else "off"}("{id}")', {"id": id}, on_done)


class HandheldTools:
    """lower_into, lift_out, squeeze, release, pour: working the held tool or container."""

    def __init__(self, robot: RobotAPI, nav: NavTools):
        self._robot = robot
        self._nav = nav

    def lower_into(self, id: str, wait: bool = True) -> ToolResult:
        return self._use(LOWER_INTO, id, wait, lambda entry: self._nav.tool_use.update(lowered_into=entry.map_name or entry.id))

    def lift_out(self, wait: bool = True) -> ToolResult:
        return self._use(LIFT_OUT, None, wait, lambda _: self._nav.tool_use.update(lowered_into=None))

    def squeeze(self, wait: bool = True) -> ToolResult:
        def squeezed(_) -> None:
            nav = self._nav
            nav.tool_use.update(squeezing=True)
            # Whatever was in the tool was used up (the egg in the cracker): forget it.
            for obj, fact in list(nav.kitchen.where.items()):
                if fact.value == nav.held["id"]:
                    nav.kitchen.forget(obj)
                    nav._map.remove(obj)

        return self._use(SQUEEZE, None, wait, squeezed)

    def release(self, wait: bool = True) -> ToolResult:
        return self._use(RELEASE, None, wait, lambda _: self._nav.tool_use.update(squeezing=False))

    def stir(self, seconds: float, wait: bool = True) -> ToolResult:
        if seconds <= 0 or seconds > 300:
            return ToolResult(False, "invalid_arguments", "seconds must be between 0 and 300")
        return self._use(STIR, None, wait, lambda _: None, seconds)

    def pour(self, id: str, wait: bool = True) -> ToolResult:
        def poured(entry: DetectionEntry) -> None:
            nav = self._nav
            target = entry.map_name or entry.id
            for obj, fact in list(nav.kitchen.where.items()):
                if fact.value == nav.held["id"]:
                    nav.kitchen.put(obj, target, YOU, nav.now())

        return self._use(POUR_INTO, id, wait, poured)

    def _use(self, action: str, id: str | None, wait: bool, on_success, seconds: float | None = None) -> ToolResult:
        nav = self._nav
        if nav.held is None:
            return ToolResult(False, MotionStatus.NOT_HOLDING, "The gripper is not holding anything.")
        entry = None
        if id is not None:
            entry = nav.visible_entry(id)
            if isinstance(entry, ToolResult):
                return entry

        def on_done(result: MotionResult) -> None:
            if result.status == MotionStatus.SUCCESS:
                on_success(entry)

        self._robot.use_tool(action, _detection(entry) if entry is not None else None, seconds)
        what = f'{action}("{id}")' if id is not None else f"{action}({seconds:g})" if seconds else f"{action}()"
        return nav.motion_tool_result(wait, what, {"id": id} if id else {}, on_done)


def _detection(entry: DetectionEntry) -> Detection:
    return Detection(entry.label, entry.target_world, entry.confidence)
