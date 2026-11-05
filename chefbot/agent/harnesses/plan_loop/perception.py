"""The PERCEPTION block and its freshness (AGENT_SPEC §9.5).

Keeps the latest detect / nearby result out of the rolling window and says how
far it can still be trusted, from tool calls since and how far the base moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from chefbot.interface.types import Pose2D, ToolResult

FRESH, STALE, CLEARED, NONE = "fresh", "possibly stale", "cleared", "none"


@dataclass(frozen=True)
class FreshnessConfig:
    stale_after_calls: int = 5
    stale_move_m: float = 0.05
    stale_turn_deg: float = 5.0
    clear_move_m: float = 0.5
    clear_turn_deg: float = 30.0


class Perception:
    def __init__(self, config: FreshnessConfig = FreshnessConfig()):
        self._cfg = config
        self.source: str | None = None  # 'detect(["egg"])' or 'nearby()'
        self.visible: list[dict] = []
        self.not_visible: list[dict] = []
        self.pose: Pose2D | None = None
        self.calls_since = 0

    def update(self, tool: str, args: dict, result: ToolResult, pose: Pose2D) -> None:
        """After every dispatched tool call."""
        if tool == "detect" and result.success:
            classes = args.get("classes")
            self.source = f"detect({_short(classes)})" if classes else "detect()"
            self.visible = [dict(d) for d in result.data.get("detections", [])]
            self.not_visible = []
            self.pose, self.calls_since = pose, 0
            return
        if tool == "nearby" and result.success:
            self.source = "nearby()"
            self.visible = [dict(d) for d in result.data.get("visible", [])]
            self.not_visible = [dict(d) for d in result.data.get("not_visible", [])]
            self.pose, self.calls_since = pose, 0
            return
        self.calls_since += 1
        if tool == "pick" and result.success:
            picked = args.get("id")
            self.visible = [d for d in self.visible if d.get("id") != picked]

    def freshness(self, pose: Pose2D) -> tuple[str, str]:
        """(state, why) for the block's header."""
        if self.source is None:
            return NONE, ""
        cfg = self._cfg
        moved = math.hypot(pose.x - self.pose.x, pose.y - self.pose.y)
        turned = abs(math.degrees(math.atan2(math.sin(pose.yaw - self.pose.yaw), math.cos(pose.yaw - self.pose.yaw))))
        if moved > cfg.clear_move_m or turned > cfg.clear_turn_deg:
            return CLEARED, "moved since"
        if moved > cfg.stale_move_m or turned > cfg.stale_turn_deg:
            return STALE, "moved since"
        if self.calls_since > cfg.stale_after_calls:
            return STALE, f"{self.calls_since} calls ago"
        return FRESH, ""

    def block(self, pose: Pose2D, held: str | None = None) -> str:
        state, why = self.freshness(pose)
        if state == NONE:
            return "PERCEPTION  (nothing yet: call detect or nearby)"
        if state == CLEARED:
            return f"PERCEPTION  (latest: {self.source} · cleared: the robot has moved; call detect or nearby)"
        header = f"PERCEPTION  (latest: {self.source} · {state}" + (f": {why})" if why else ")")
        visible = _ids(self.visible) + (f", {held} (held)" if held and self.visible else f"{held} (held)" if held else "")
        lines = [header, "  visible:      " + (visible or "nothing")]
        if self.source == "nearby()":
            lines.append("  not visible:  " + (_ids(self.not_visible) or "nothing"))
        return "\n".join(lines)


def _ids(items: list[dict]) -> str:
    def one(d: dict) -> str:
        notes = [f"on {d['on']}"] if d.get("on") else []
        notes += ["in the gripper"] if d.get("in_gripper") else []
        notes += [d["state"]] if d.get("state") else []
        return d.get("id", "?") + (f" ({', '.join(notes)})" if notes else "")

    return ", ".join(one(d) for d in items)


def _short(classes) -> str:
    return "[" + ", ".join(f'"{c}"' for c in classes) + "]"
