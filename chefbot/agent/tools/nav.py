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
from chefbot.agent.world.map import MapEntry, WorldMap
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

    # -- go_to ---------------------------------------------------------------

    def go_to(self, place: str, wait: bool = True) -> ToolResult:
        entry = self._map.get(place)
        if entry is None:
            known = ", ".join(e.name for e in self._map.list())
            return ToolResult(False, "unknown_place", f'"{place}" is not on the map. Known places: {known}.')
        self._robot.go_to_pose(self._approach_pose(entry))
        return self._motion_tool_result(wait, f'go_to("{place}")', {"place": place})

    def _approach_pose(self, entry: MapEntry) -> Pose2D:
        # An object stored inside a fixture is approached the way the fixture is.
        if entry.facing is None and entry.inside and (holder := self._map.get(entry.inside)):
            entry = holder
        location = np.array(entry.location)
        if entry.facing is not None:
            direction = np.array(entry.facing)
        else:
            direction = self._robot.get_base_pose().xy - location
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        goal = location + self._cfg.approach_distance * direction
        return Pose2D(float(goal[0]), float(goal[1]), math.atan2(-direction[1], -direction[0]))

    # -- detect --------------------------------------------------------------

    def detect(self, classes: list[str] | None = None) -> ToolResult:
        requested = _unique(classes if classes else self._cfg.default_vocab)
        pose = self._robot.get_base_pose()
        in_range = self._map.in_range(pose, self._cfg.nearby_radius)
        query = _unique(requested + [e.label for e in in_range])
        raw = self._detector.detect(query)

        entries = self._build_entries(raw, in_range, pose)
        self.detections.replace(entries)

        found = sorted(
            (e for e in entries if e.label in requested), key=lambda e: e.distance_to(pose)
        )
        data = {"detections": [{"id": e.id, "label": e.label} for e in found]}
        if not found:
            return ToolResult(True, None, f"Nothing found for: {', '.join(requested)}.", data)
        return ToolResult(True, None, f"Found: {', '.join(e.id for e in found)}.", data)

    def _build_entries(
        self, raw: list[Detection], in_range: list[MapEntry], pose: Pose2D
    ) -> list[DetectionEntry]:
        now = time.time()

        def entry(det: Detection, id: str, map_name: str | None) -> DetectionEntry:
            position = np.asarray(det.position_world, dtype=float)
            return DetectionEntry(id, det.label, position, float(det.confidence), pose, now, map_name)

        map_labels = self._map.labels()
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

    def _nearby(self) -> NearbyView:
        return build_nearby(
            self._map, self.detections, self._robot.get_base_pose(), self._cfg.nearby_radius
        )

    def nearby(self) -> ToolResult:
        view = self._nearby()
        data = {
            "visible": [{"id": e.id, "label": e.label} for e in view.visible],
            "not_visible": [{"id": e.name, "label": e.label} for e in view.not_visible],
        }
        return ToolResult(True, None, view.describe(), data)

    # -- align ---------------------------------------------------------------

    def align(self, id: str, wait: bool = True) -> ToolResult:
        view = self._nearby()
        entry = view.find_visible(id)
        if entry is None:
            if view.find_not_visible(id) is not None:
                return ToolResult(
                    False,
                    "not_visible",
                    f'"{id}" is nearby but the last detect did not see it. '
                    f'go_to("{id}") first, then detect again.',
                )
            hint = "Call detect first." if view.is_empty() else view.describe()
            return ToolResult(False, "unknown_id", f'"{id}" is not in the nearby view. {hint}')

        pose = self._robot.get_base_pose()
        if entry.distance_to(pose) > self._cfg.max_align_distance:
            return ToolResult(
                False, "too_far", f'"{id}" is too far away to align to. Use go_to to get closer first.'
            )

        self._robot.align_to_point(entry.position_world)
        result = self._motion_tool_result(wait, f'align("{id}")', {"id": id})
        if wait and result.success:
            result.data["refined"] = self._refine(entry)
        return result

    def _refine(self, entry: DetectionEntry) -> bool:
        """Re-detect the item from the aligned pose and update only its entry."""
        candidates = [d for d in self._detector.detect([entry.label]) if d.label == entry.label]
        if not candidates:
            return False
        best = min(candidates, key=lambda d: np.linalg.norm(d.position_world - entry.position_world))
        if np.linalg.norm(best.position_world - entry.position_world) > self._cfg.refine_radius:
            return False
        self.detections.refine(
            entry.id, best.position_world, float(best.confidence), self._robot.get_base_pose(), time.time()
        )
        return True

    # -- motion status ---------------------------------------------------------

    def wait(self, timeout_s: float | None = None) -> ToolResult:
        return self._from_motion(self._robot.wait(timeout_s), "wait")

    def get_status(self) -> ToolResult:
        return self._from_motion(self._robot.get_status(), "get_status")

    def stop(self) -> ToolResult:
        self._robot.stop_base()
        pose = self._robot.get_base_pose()
        return ToolResult(True, None, "Base stopped.", {"status": MotionStatus.STOPPED, "pose": pose_data(pose)})

    def get_pose(self) -> ToolResult:
        pose = self._robot.get_base_pose()
        return ToolResult(True, None, "", {"pose": pose_data(pose)})

    def _motion_tool_result(self, wait: bool, what: str, data: dict) -> ToolResult:
        if wait:
            return self._from_motion(self._robot.wait(), what, data)
        return ToolResult(
            True,
            None,
            f"{what} started. Call wait or get_status to follow it, stop to cancel.",
            {**data, "status": MotionStatus.MOVING},
        )

    def _from_motion(self, result: MotionResult, what: str, data: dict | None = None) -> ToolResult:
        data = {**(data or {}), "status": result.status}
        if result.final_pose is not None:
            data["pose"] = pose_data(result.final_pose)
        ok = result.status in _OK_STATUSES
        message = f"{what}: {result.status}" + (f" ({result.message})" if result.message else "")
        return ToolResult(ok, None if ok else result.status, message, data)


def _unique(items) -> list[str]:
    return list(dict.fromkeys(items))
