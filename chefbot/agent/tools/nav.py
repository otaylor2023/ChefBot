"""Navigation tools: go_to, detect, nearby, align, wait, get_status, stop, get_pose.

The tools choose goals (poses, object ids); the robot layer decides how to get
there. Nothing here knows the robot's geometry or which env it is running in.
"""

from __future__ import annotations

import math
import time

import numpy as np

from chefbot.agent.config import AgentConfig
from chefbot.agent.world.detections import DetectionEntry, Detections
from chefbot.agent.world.kitchen import MAP, SEEN, YOU, KitchenBelief
from chefbot.agent.world.map import MapEntry, WorldMap, label_of
from chefbot.agent.world.nearby import NearbyView, build_nearby
from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import Detection, MotionResult, MotionStatus, Pose2D, ToolResult

# Motion statuses that are not failures from the agent's point of view.
_OK_STATUSES = (MotionStatus.SUCCESS, MotionStatus.MOVING, MotionStatus.IDLE)


def pose_data(pose: Pose2D) -> dict:
    """Poses cross the LLM boundary in degrees."""
    return {"x": round(pose.x, 3), "y": round(pose.y, 3), "yaw_deg": round(math.degrees(pose.yaw), 1)}


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class NavTools:
    def __init__(self, robot: RobotAPI, detector: Detector, world_map: WorldMap, config: AgentConfig):
        self._robot = robot
        self._detector = detector
        self._map = world_map
        self._cfg = config
        self.detections = Detections()
        self.held: dict | None = None  # {"id", "label"} of the object in the gripper
        # Bookkeeping to run when the motion in progress ends (set by tools started with wait=False too).
        self._on_done = None
        # Set by the agent loop (TimeTools): blocking waits wake on events.
        self.time = None
        # Map objects looked for close to where the map has them, and not seen: name -> times.
        self.not_found: dict[str, int] = {}
        # Things asked for and not found: label -> map fixtures that were in view each time.
        self.searched: dict[str, set[str]] = {}
        # The held tool: what it is lowered into, and whether its trigger is squeezed.
        self.tool_use: dict = {"lowered_into": None, "squeezing": False}
        # Door / burner states and where objects are (KITCHEN block); run time from `now`.
        self.kitchen = KitchenBelief()
        self.now = time.monotonic  # the agent loop sets its run clock
        for e in world_map.list():
            if e.kind == "object" and e.inside:
                self.kitchen.put(e.name, e.inside, MAP, 0.0)

    # -- go_to ---------------------------------------------------------------

    def go_to(self, place: str, wait: bool = True) -> ToolResult:
        entry = self._map.get(place)
        if entry is None:
            known = ", ".join(e.name for e in self._map.list())
            return ToolResult(False, "unknown_place", f'"{place}" is not on the map. Known places: {known}.')
        self.tool_use["lowered_into"] = None  # the robot lifts a lowered tool out before driving
        self._robot.go_to_pose(self._approach_pose(entry))
        result = self.motion_tool_result(wait, f'go_to("{place}")', {"place": place})
        if entry.kind == "object" and result.success:
            where = self.describe_where(place)
            if where:
                result.message += f"; {where}"
                result.data["where"] = where
        return result

    def _approach_pose(self, entry: MapEntry) -> Pose2D:
        # An object on or in a fixture is approached from the fixture's open side,
        # standing in front of the object itself (a counter can be meters long).
        facing = entry.facing
        if facing is None and entry.inside and (holder := self._map.get(entry.inside)):
            facing = holder.facing
        location = np.array(entry.location)
        if facing is not None:
            direction = np.array(facing)
        else:
            direction = self._robot.get_base_pose().xy - location
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        goal = location + self._cfg.approach_distance * direction
        return Pose2D(float(goal[0]), float(goal[1]), math.atan2(-direction[1], -direction[0]))

    # -- detect --------------------------------------------------------------

    def detect(self, classes: list[str] | None = None) -> ToolResult:
        objects = sorted({e.label for e in self._map.list() if e.kind == "object"})
        # Classes are labels; a map name works too ("egg_cracker" -> "egg cracker").
        asked = [c.replace("_", " ") for c in classes] if classes else None
        requested = _unique(asked if asked else [*self._cfg.default_vocab, *objects])
        pose = self._robot.get_base_pose()
        in_range = self._map.in_range(pose, self._cfg.nearby_radius)
        query = _unique(requested + [e.label for e in in_range])
        raw = self._detector.detect(query)

        entries = self._build_entries(raw, in_range, pose)
        self._name_held_contents(entries)
        self.detections.replace(entries)

        found = sorted(
            (e for e in entries if e.label in requested), key=lambda e: e.distance_to(pose)
        )
        seen = {e.id for e in entries}
        for m in in_range:
            if m.kind != "object" or m.label not in requested:
                continue
            if m.name in seen:
                self.not_found.pop(m.name, None)
            elif _in_front(pose, m):  # it should have been in view
                self.not_found[m.name] = self.not_found.get(m.name, 0) + 1
        fixtures = {m.name for m in in_range if m.kind == "fixture" and m.label != "burner" and _in_front(pose, m)}
        found_labels = {e.label for e in found}
        # An empty detect looks for everything it knows: with search_default_vocab off, none of
        # that is something you are searching for (only what you ask for by name is).
        tracked = set(requested) if asked or self._cfg.search_default_vocab else set()
        for label in requested:
            if label in found_labels:
                self.searched.pop(label, None)
            elif label not in tracked:
                continue
            elif label not in self._map.labels() or any(
                m.kind == "object" and m.label == label for m in self._map.list()
            ):  # a thing, not a kind of fixture
                self.searched.setdefault(label, set()).update(fixtures)
        described = {e.id: _described(e, entries) for e in entries}
        self._learn(entries, described)
        data = {"detections": [described[e.id] for e in found]}
        if not found:
            return ToolResult(True, None, f"Nothing found for: {', '.join(requested)}.", data)
        return ToolResult(True, None, f"Found: {', '.join(e.id for e in found)}.", data)

    def _build_entries(
        self, raw: list[Detection], in_range: list[MapEntry], pose: Pose2D
    ) -> list[DetectionEntry]:
        now = time.time()

        def entry(det: Detection, id: str, map_name: str | None) -> DetectionEntry:
            position = np.asarray(det.position_world, dtype=float)
            near = None if det.near_point_world is None else np.asarray(det.near_point_world, dtype=float)
            return DetectionEntry(
                id, det.label, position, float(det.confidence), pose, now, map_name, near,
                getattr(det, "state", None), bool(getattr(det, "in_gripper", False)),
            )  # fmt: skip

        # Fixtures stay where the map says; loose objects move, so an object seen
        # away from its map location is still reported (under a fresh id).
        map_labels = {e.label for e in self._map.list() if e.kind == "fixture"}
        # Map matching: closest (detection, map entry) pairs first, one-to-one.
        pairs = sorted(
            (_xy_dist(det.position_world, m.location), i, m.name)
            for i, det in enumerate(raw)
            for m in in_range
            if m.label == det.label
        )
        matched: dict[int, str] = {}
        for dist, i, name in pairs:
            if dist > self._cfg.map_match_radius:
                break
            if i not in matched and name not in matched.values():
                matched[i] = name

        entries = [entry(raw[i], name, name) for i, name in matched.items()]
        # Unmatched detections of map labels are dropped: those are the map's to report.
        others = sorted(
            (det for i, det in enumerate(raw) if i not in matched and det.label not in map_labels),
            key=lambda det: _xy_dist(det.position_world, (pose.x, pose.y)),
        )
        counts: dict[str, int] = {}
        for det in others:
            counts[det.label] = counts.get(det.label, 0) + 1
            entries.append(entry(det, f"{det.label}_{counts[det.label]}", None))
        return entries

    # -- nearby --------------------------------------------------------------

    # -- what the agent believes about the kitchen (KITCHEN block) ---------------

    def _learn(self, entries: list[DetectionEntry], described: dict[str, dict]) -> None:
        """Fold a detect into the belief: states of doors and burners, and where
        objects are. An object seen that the map does not have joins the map."""
        now = self.now()
        fixture_labels = {e.label for e in self._map.list() if e.kind == "fixture"}
        for e in entries:
            if e.state is not None:
                name = e.map_name or self.fixture_of_handle(e)
                if name is not None:
                    self.kitchen.set_state(name, e.state, SEEN, now)
        for e in entries:
            if e.label in fixture_labels or e.label in _PARTS or e.label.endswith(" handle"):
                continue  # fixtures and their parts are not loose objects
            if e.in_gripper:
                if self.held is not None:
                    self.kitchen.put(e.map_name or e.id, self.held["id"], SEEN, now)
                continue
            on = described[e.id].get("on")
            holder_entry = next((x for x in entries if x.id == on), None) if on else None
            known = self._map.get(e.map_name) if e.map_name else None
            if holder_entry is not None:
                holder = holder_entry.map_name or holder_entry.id
            elif known is not None and known.inside and _xy_dist(e.position_world, known.location) <= _SAME_SPOT:
                holder = self.kitchen.where[e.map_name].value if e.map_name in self.kitchen.where else known.inside
            else:
                holder = self._nearest_fixture(e.position_world)
            if holder is None:
                continue
            name = e.map_name or e.id
            if e.map_name is None and self._map.get(name) is not None:
                # A new one whose detection id a mapped one (seen elsewhere, out of range) already has.
                n = 1
                while self._map.get(f"{e.label}_{n}") is not None:
                    n += 1
                name = f"{e.label}_{n}"
            mapped = self._map.get(name)
            if (e.map_name is None or mapped.inside != self._top(holder)
                    or _xy_dist(e.position_world, mapped.location) > _MOVED):  # fmt: skip
                # Seen somewhere the map does not have it: the map follows.
                self._map.put(MapEntry(name, (float(e.position_world[0]), float(e.position_world[1])),
                                       kind="object", inside=self._top(holder)))  # fmt: skip
                e.map_name = name
            old = self.kitchen.where.get(name)
            if old is None or old.value != holder:
                self.kitchen.put(name, holder, SEEN, now)

    def _name_held_contents(self, entries: list[DetectionEntry]) -> None:
        """Things seen inside the held tool keep the names you know them by (egg_2 in the cracker)."""
        held = self.held["id"] if self.held else None
        inside = [o for o, f in self.kitchen.where.items() if held and f.value == held]
        for e in entries:
            if not e.in_gripper:
                continue
            match = next((o for o in inside if label_of(o) == e.label), None)
            if match is not None:
                e.id, e.map_name = match, None
                inside.remove(match)

    def held_contents(self, obj: str) -> list[str]:
        """Everything believed to be in or on `obj`, at any depth."""
        out, todo = [], [obj]
        while todo:
            holder = todo.pop()
            for name, fact in self.kitchen.where.items():
                if fact.value == holder and name not in out:
                    out.append(name)
                    todo.append(name)
        return out

    def record_place(self, obj: str, target: DetectionEntry) -> None:
        """You put `obj` down on / in `target`: the map has it there now."""
        holder = target.map_name or target.id
        xy = target.target_world
        self._map.put(MapEntry(obj, (float(xy[0]), float(xy[1])), kind="object", inside=self._top(holder)))
        self.kitchen.put(obj, holder, YOU, self.now())
        for inner in self.held_contents(obj):  # what was in it comes back with it
            self._map.put(MapEntry(inner, (float(xy[0]), float(xy[1])), kind="object", inside=self._top(holder)))

    def record_pick(self, obj: str) -> None:
        self.kitchen.forget(obj)
        for inner in self.held_contents(obj):  # its contents go with it: off the map, still believed inside
            self._map.remove(inner)

    def record_state(self, target: DetectionEntry, state: str) -> None:
        """You opened / closed / turned on / turned off `target` (a fixture, its handle, a burner)."""
        name = target.map_name or self.fixture_of_handle(target)
        if name is None:
            # A door that is not on the map (one of many cabinets): it joins the map, so
            # you can go_to it and close it again later.
            base = target.label.removesuffix(" handle").replace(" ", "_")
            n = 1
            while self._map.get(f"{base}_{n}") is not None:
                n += 1
            name = f"{base}_{n}"
            xy = target.position_world
            self._map.put(MapEntry(name, (float(xy[0]), float(xy[1]))))
        self.kitchen.set_state(name, state, YOU, self.now())

    def describe_where(self, obj: str) -> str:
        fact = self.kitchen.where.get(obj)
        if fact is None:
            return ""
        on = "in" if _is_container(fact.value) else "on"
        ago = _ago(self.now() - fact.t)
        if fact.source == YOU:
            return f"you put the {label_of(obj)} {on} {fact.value} {ago} ago"
        if fact.source == SEEN:
            return f"the {label_of(obj)} was seen {on} {fact.value} {ago} ago"
        return f"the map has the {label_of(obj)} {on} {fact.value}"

    def fixture_of_handle(self, entry: DetectionEntry) -> str | None:
        """The map fixture a handle (or another fixture part) belongs to."""
        base = entry.label.removesuffix(" handle")
        candidates = [m for m in self._map.list() if m.kind == "fixture" and m.label == base]
        if not candidates:
            return None
        best = min(candidates, key=lambda m: _xy_dist(entry.position_world, m.location))
        return best.name if _xy_dist(entry.position_world, best.location) <= 1.5 else None

    def _nearest_fixture(self, position) -> str | None:
        fixtures = [m for m in self._map.list() if m.kind == "fixture"]
        if not fixtures:
            return None
        # Over a fixture's footprint: that fixture (the first on the map, as the map builder decides).
        over = next((m for m in fixtures if m.contains(float(position[0]), float(position[1]))), None)
        if over is not None:
            return over.name
        best = min(fixtures, key=lambda m: _xy_dist(position, m.location))
        return best.name if _xy_dist(position, best.location) <= 1.0 else None

    def _top(self, name: str) -> str:
        """The map fixture a place is on or in (an object's holder, followed down)."""
        seen = set()
        while name not in seen:
            seen.add(name)
            entry = self._map.get(name)
            if entry is None or entry.kind == "fixture":
                return name
            fact = self.kitchen.where.get(name)
            name = fact.value if fact is not None else (entry.inside or name)
        return name

    def missing(self) -> set[str]:
        """Map objects not found at their map spot twice or more: the map is wrong about them."""
        return {name for name, times in self.not_found.items() if times >= _MISSING_AFTER}

    def _nearby(self) -> NearbyView:
        view = build_nearby(self._map, self.detections, self._robot.get_base_pose(), self._cfg.nearby_radius)
        gone = self.missing()
        if gone:
            view.not_visible = [e for e in view.not_visible if e.name not in gone]
        return view

    def nearby(self) -> ToolResult:
        view = self._nearby()
        data = {
            "visible": [{"id": e.id, "label": e.label} for e in view.visible],
            "not_visible": [{"id": e.name, "label": e.label} for e in view.not_visible],
            "holding": dict(self.held) if self.held else None,
        }
        message = view.describe()
        if self.held:
            message += f" Holding: {self.held['id']}."
        return ToolResult(True, None, message, data)

    # -- align ---------------------------------------------------------------

    def align(self, id: str, wait: bool = True) -> ToolResult:
        entry = self.visible_entry(id)
        if isinstance(entry, ToolResult):
            return entry

        pose = self._robot.get_base_pose()
        if _xy_dist(entry.target_world, (pose.x, pose.y)) > self._cfg.max_align_distance:
            return ToolResult(
                False, "too_far", f'"{id}" is too far away to align to. Use go_to to get closer first.'
            )

        self._robot.align_to_point(entry.target_world)
        result = self.motion_tool_result(wait, f'align("{id}")', {"id": id})
        if wait and result.success:
            result.data["refined"] = self._refine(entry)
        return result

    def visible_entry(self, id: str) -> DetectionEntry | ToolResult:
        """The visible item `id` from the nearby view, or the error to return instead."""
        view = self._nearby()
        entry = view.find_visible(id)
        if entry is not None:
            return entry
        if view.find_not_visible(id) is not None:
            return ToolResult(
                False,
                "not_visible",
                f'"{id}" is nearby but the last detect did not see it. '
                f'go_to("{id}") first, then detect again.',
            )
        hint = "Call detect first." if view.is_empty() else view.describe()
        return ToolResult(False, "unknown_id", f'"{id}" is not in the nearby view. {hint}')

    def _refine(self, entry: DetectionEntry) -> bool:
        """Re-detect the item from the aligned pose and update only its entry."""
        candidates = [d for d in self._detector.detect([entry.label]) if d.label == entry.label]
        if not candidates:
            return False
        best = min(candidates, key=lambda d: np.linalg.norm(d.position_world - entry.position_world))
        if np.linalg.norm(best.position_world - entry.position_world) > self._cfg.refine_radius:
            return False
        self.detections.refine(
            entry.id,
            best.position_world,
            float(best.confidence),
            self._robot.get_base_pose(),
            time.time(),
            best.near_point_world,
        )
        return True

    # -- motion status ---------------------------------------------------------

    def wait(self, timeout_s: float | None = None) -> ToolResult:
        if self.time is not None:
            return self.time.wait(timeout_s, finish=lambda result: self._from_motion(result, "wait"))
        return self._from_motion(self._robot.wait(timeout_s), "wait")

    def get_status(self) -> ToolResult:
        return self._from_motion(self._robot.get_status(), "get_status")

    def stop(self) -> ToolResult:
        self._robot.stop()
        self._finish(self._robot.get_status())
        pose = self._robot.get_base_pose()
        return ToolResult(
            True, None, "Stopped the base and the arm.", {"status": MotionStatus.STOPPED, "pose": pose_data(pose)}
        )

    def get_pose(self) -> ToolResult:
        pose = self._robot.get_base_pose()
        return ToolResult(True, None, "", {"pose": pose_data(pose)})

    def motion_tool_result(self, wait: bool, what: str, data: dict, on_done=None) -> ToolResult:
        """Result of a tool that just started a motion. `on_done(MotionResult)`
        runs once, when the motion's final result is first seen."""
        self._on_done = on_done
        return self._motion_tool_result(wait, what, data)

    def _finish(self, result: MotionResult) -> None:
        if result.status == MotionStatus.MOVING:
            return
        on_done, self._on_done = self._on_done, None
        if on_done is not None:
            on_done(result)
        elif result.status == MotionStatus.GRASP_FAILED:
            self.held = None  # dropped while driving

    def _motion_tool_result(self, wait: bool, what: str, data: dict) -> ToolResult:
        if wait and self.time is not None:
            result, events, why = self.time.await_events()
            if why != "event":
                return self._from_motion(result, what, data)
            # Woken by an event: the motion keeps going.
            self.time.note_motion(what, True)
            return ToolResult(
                True, None,
                f"{what} is still running; woke for: " + "; ".join(e.message for e in events)
                + ". Call wait to keep waiting for it, or stop to cancel it.",
                {**data, "status": MotionStatus.MOVING, "events": [e.to_dict() for e in events]},
            )  # fmt: skip
        if wait:
            return self._from_motion(self._robot.wait(), what, data)
        if self.time is not None:
            self.time.note_motion(what, True)
        return ToolResult(
            True,
            None,
            f"{what} started. Call wait or get_status to follow it, stop to cancel.",
            {**data, "status": MotionStatus.MOVING},
        )

    def _from_motion(self, result: MotionResult, what: str, data: dict | None = None) -> ToolResult:
        self._finish(result)
        data = {**(data or {}), **result.data, "status": result.status}
        if result.final_pose is not None:
            data["pose"] = pose_data(result.final_pose)
        ok = result.status in _OK_STATUSES
        message = f"{what}: {result.status}" + (f" ({result.message})" if result.message else "")
        return ToolResult(ok, None if ok else result.status, message, data)


_LOOKED_CLOSE = 2.0  # m: a map object this close that a detect asked for and did not see is "not found"
_MISSING_AFTER = 2  # not found at its map spot this many times: treated as missing
_SEARCH_RANGE = 1.5  # m: a fixture this close and in front counts as searched by a detect
_SEARCH_HALF_ANGLE = math.radians(60)


def _in_front(pose: Pose2D, entry: MapEntry) -> bool:
    dx, dy = entry.location[0] - pose.x, entry.location[1] - pose.y
    if math.hypot(dx, dy) > _SEARCH_RANGE:
        return False
    off = math.atan2(dy, dx) - pose.yaw
    return abs(math.atan2(math.sin(off), math.cos(off))) <= _SEARCH_HALF_ANGLE
_HOLDERS = {"burner", "pan", "pot", "plate", "bowl", "toaster", "cutting board", "mug"}
_ON_RADIUS = 0.1  # m: an object this close (xy) above a holder is on / in it


def _described(entry, entries) -> dict:
    """A detection for the agent: id, label, and what it sits on or in, from the positions."""
    out = {"id": entry.id, "label": entry.label}
    if entry.state is not None:
        out["state"] = entry.state
    if entry.in_gripper:
        out["in_gripper"] = True
        return out
    if entry.label == "burner":
        return out
    p = entry.position_world
    under = [
        e for e in entries
        if e is not entry and e.label in _HOLDERS and e.label != entry.label
        and float(np.linalg.norm(e.position_world[:2] - p[:2])) <= _ON_RADIUS and e.position_world[2] <= p[2]
    ]  # fmt: skip
    if under:
        out["on"] = max(under, key=lambda e: e.position_world[2]).id
    return out


_MOVED = 0.01  # m: a sighting this far from the map's spot moves the map entry
_SAME_SPOT = 0.3  # m: an object seen this close to where you thought it was has not moved
_PARTS = {"burner", "faucet", "knob", "shelf"}  # parts of fixtures, never loose objects
_CONTAINERS = {"fridge", "cabinet", "microwave", "drawer", "oven", "toaster", "pan", "pot", "bowl", "mug", "sink", "egg cracker"}


def _is_container(name: str) -> bool:
    return label_of(name) in _CONTAINERS


def _ago(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))
