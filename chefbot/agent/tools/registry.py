"""Tool registry: function-calling schemas for the LLM, and dispatch.

Constructed with a RobotAPI and a Detector; it never knows which env is behind them.
"""

from __future__ import annotations

import inspect
from typing import Any

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.nav import NavTools
from chefbot.agent.world.map import WorldMap
from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import ToolResult

_WAIT_PARAM = {
    "type": "boolean",
    "description": (
        "true (default): block until the motion ends and return its result. "
        'false: return immediately with status "moving"; follow up with wait or '
        "get_status, or cancel with stop."
    ),
}


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "go_to",
        "Drive across the kitchen to a place on the map (a fixture such as the stove or fridge, "
        "or a mapped object). This is coarse navigation: it leaves the robot standing near the "
        "place and facing it, not within arm's reach of anything in particular. It is step 1 of "
        "the navigation workflow: go_to a fixture -> detect what you need -> nearby to see "
        "everything around you -> align to a visible id. Fails with unknown_place (the message "
        "lists the known places), no_path, blocked, timeout or stopped.",
        {
            "place": {"type": "string", "description": 'Map name, e.g. "stove", "fridge", "counter_2".'},
            "wait": _WAIT_PARAM,
        },
        ["place"],
    ),
    _tool(
        "detect",
        "Look through the robot's camera from where it stands and find instances of the given "
        "classes (parts and loose objects such as burners, handles, pans). Returns an id for each "
        'instance found, nearest first, e.g. "burner_1". An empty list means none are in view '
        "from here. Each call replaces the results of the previous one, so ids from an earlier "
        "detect are no longer valid. Call it after go_to, and again whenever the robot has "
        "moved or turned.",
        {
            "classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'What to look for, e.g. ["burner", "pan"]. Omit to use the default kitchen vocabulary.',
            }
        },
    ),
    _tool(
        "nearby",
        "List what is around the robot right now, nearest first, without looking again. "
        '"visible" items were seen by the last detect and can be passed to align. "not_visible" '
        "items are on the map and close by but were not seen (behind the robot, hidden, or "
        "missed): go_to them first, then detect again.",
    ),
    _tool(
        "align",
        "Fine positioning: move the base so the arm can reach one specific item. The id must be "
        "a visible item from nearby() (which includes everything the last detect returned). "
        "Fails with unknown_id (the message lists what is nearby), not_visible (go_to it first), "
        "too_far (go_to first), unreachable (no place to stand from which the arm reaches it), "
        "no_path, blocked, timeout or stopped.",
        {
            "id": {"type": "string", "description": 'A visible id from nearby(), e.g. "burner_1".'},
            "wait": _WAIT_PARAM,
        },
        ["id"],
    ),
    _tool(
        "wait",
        "Wait for the current motion to finish and return its result. With timeout_s, returns "
        'status "moving" if the motion is still going when the time is up (it keeps going). '
        'Returns status "idle" if no motion was ever started.',
        {
            "timeout_s": {
                "type": "number",
                "description": "Seconds to wait at most. Omit to wait until the motion ends.",
            }
        },
    ),
    _tool("get_status", "Report the status of the current or last motion without waiting."),
    _tool(
        "stop",
        "Stop the base immediately. Works at any time, including in the middle of a go_to or "
        "align that was started with wait=false.",
    ),
    _tool("get_pose", "Return the robot base's current position (meters) and heading (degrees)."),
]


class ToolRegistry:
    def __init__(self, robot: RobotAPI, detector: Detector, world_map: WorldMap, config: AgentConfig):
        self.tools = NavTools(robot, detector, world_map, config)
        self._handlers = {
            schema["function"]["name"]: getattr(self.tools, schema["function"]["name"])
            for schema in TOOL_SCHEMAS
        }

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def dispatch(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                False, "unknown_tool", f'No tool named "{name}". Tools: {", ".join(self._handlers)}.'
            )
        # A null optional argument means "use the default".
        args = {k: v for k, v in (args or {}).items() if v is not None}
        problem = _check_args(name, args)
        if problem is None:
            try:
                inspect.signature(handler).bind(**args)
            except TypeError as exc:
                problem = str(exc)
        if problem is not None:
            return ToolResult(False, "invalid_arguments", f"{name}: {problem}")
        return handler(**args)


_JSON_TYPES = {"string": str, "boolean": bool, "number": (int, float), "array": list}


def _check_args(name: str, args: dict[str, Any]) -> str | None:
    """Validate argument types against the tool's schema. Returns a problem or None."""
    schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            return f'unexpected argument "{key}"'
        expected = _JSON_TYPES[spec["type"]]
        if not isinstance(value, expected) or (spec["type"] == "number" and isinstance(value, bool)):
            return f'"{key}" must be a {spec["type"]}'
        if spec["type"] == "array" and not all(isinstance(v, str) for v in value):
            return f'"{key}" must be an array of strings'
    return None
