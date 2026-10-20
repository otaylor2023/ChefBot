"""Tool registry: function-calling schemas for the LLM, and dispatch.

The workflow the descriptions teach:
    go_to -> detect -> align -> pick -> go_to -> detect -> align -> place

Constructed with a RobotAPI and a Detector; it never knows which env is behind them.
"""

from __future__ import annotations

import inspect
from typing import Any

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.manip import HandheldTools, KitchenTools, ManipTools
from chefbot.agent.tools.nav import NavTools
from chefbot.agent.world.map import WorldMap
from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import (
    APPLIANCES,
    ARTICULATE,
    BASE,
    GRASP,
    HANDHELD_TOOLS,
    POUR,
    TRIGGER_TOOLS,
    RobotAPI,
)
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
        "the workflow for moving something: go_to a fixture -> detect what you need -> align to "
        "it -> pick it -> go_to where it goes -> detect -> align to the spot -> place. (nearby "
        "lists everything around you at any point.) The arm tucks itself in before the base "
        "moves, and keeps hold of whatever it is carrying. Fails with unknown_place (the message "
        "lists the known places), no_path, blocked, timeout or stopped; grasp_failed means the "
        "carried object was dropped on the way.",
        {
            "place": {"type": "string", "description": 'Map name, e.g. "stove", "fridge", "counter_2".'},
            "part": {"type": "string", "enum": ["left", "right"],
                     "description": "Optional, for a long fixture (a counter or island): stand in front of "
                                    "its left or right end, as you face it, instead of its middle."},
            "along": {"type": "number", "minimum": -1, "maximum": 1,
                      "description": "Optional, finer than part: where along a long fixture to stand, "
                                     "-1 its left end, 0 its middle, 1 its right end."},  # fmt: skip
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
        "items are on the map and close by but were not seen (behind the robot, hidden, for "
        "example inside a closed fridge, or missed): go_to them first, then detect again. Also "
        "says what the gripper is holding.",
    ),
    _tool(
        "align",
        "Fine positioning: move the base so the arm can reach one specific item. Do this before "
        "pick, place, open or close. The id must be "
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
        "Stop the base and the arm immediately. Works at any time, including in the middle of a "
        "motion that was started with wait=false. An object in the gripper stays in the gripper.",
    ),
    _tool("get_pose", "Return the robot base's current position (meters) and heading (degrees)."),
    _tool(
        "pick",
        "Pick up one visible object with the gripper and lift it. Align to the object first: pick "
        "does not move the base. Only one object can be held at a time. Fails with unknown_id, "
        "not_visible, already_holding (place what you hold first), no_grasp (the id is not "
        "something the gripper can pick up, such as a counter), unreachable (align to it first), "
        "grasp_failed (the gripper closed on nothing or the object slipped; detect and try "
        "again), collision (the arm touched something and stopped where it was), stopped or "
        "timeout.",
        {
            "id": {"type": "string", "description": 'A visible id from nearby(), e.g. "egg_1" or "mug".'},
            "wait": _WAIT_PARAM,
        },
        ["id"],
    ),
    _tool(
        "place",
        "Set the held object down on a visible surface or into a visible container: a counter, a "
        "burner, a pan, a plate, a cutting board. The id names where it goes, not the object. "
        "Align to that id first: place does not move the base. Fails with unknown_id, "
        "not_visible, not_holding, no_placement (no free spot there), unreachable (align to it "
        "first), collision (the arm touched something and stopped; the object is still held), "
        "stopped or timeout.",
        {
            "id": {"type": "string", "description": 'Where to put it: a visible id, e.g. "counter_1", "burner_2", "pan".'},
            "wait": _WAIT_PARAM,
        },
        ["id"],
    ),
    _tool(
        "open",
        "Open a fixture: a fridge, microwave, dishwasher, oven, cabinet or drawer, named by its "
        "own id or the id of its handle. Align to the fixture or its handle first. Opening is "
        'scripted for now (the result says "scripted": true): the door moves by itself, and the '
        "robot steps back out of its way if it has to, so detect again afterwards. An open door "
        "blocks the floor in front of it until it is closed. Fails with unknown_id, not_visible, "
        "unreachable (align to it first), unsupported (the id is not something that opens), "
        "blocked or stopped.",
        {
            "id": {"type": "string", "description": 'A visible id, e.g. "fridge" or "fridge handle_1".'},
            "wait": _WAIT_PARAM,
        },
        ["id"],
    ),
    _tool(
        "close",
        "Close a fixture that open works on. Same rules and errors as open.",
        {
            "id": {"type": "string", "description": 'A visible id, e.g. "fridge" or "fridge handle_1".'},
            "wait": _WAIT_PARAM,
        },
        ["id"],
    ),
    _tool(
        "turn_on",
        "Turn on a burner or an appliance (toaster, microwave, kettle, coffee machine), named "
        "by its visible id. Align to it first. Scripted for now (\"scripted\": true). Fails "
        "with unknown_id, not_visible, unreachable (align to it first), unsupported (not "
        "something that turns on), or stopped.",
        {"id": {"type": "string", "description": 'A visible id, e.g. "burner_2" or "toaster".'}, "wait": _WAIT_PARAM},
        ["id"],
    ),
    _tool(
        "turn_off",
        "Turn off a burner or an appliance. Same rules and errors as turn_on.",
        {"id": {"type": "string", "description": 'A visible id, e.g. "burner_2".'}, "wait": _WAIT_PARAM},
        ["id"],
    ),
    _tool("holding", "Say what the gripper is holding, if anything."),
]


# Handheld tools and pouring (offered when the robot has HANDHELD_TOOLS / POUR).
HANDHELD_SCHEMAS: list[dict] = [
    _tool(
        "lower_into",
        "Lower the tool or container you are holding into (or just over) a visible container: "
        "a whisk into a bowl, an egg cracker over a bowl or pan. Align to the container first. "
        "It stays there until lift_out, or until you drive off (it is lifted out first). Fails "
        "with not_holding, unknown_id, not_visible, unreachable (align first) or no_placement.",
        {"id": {"type": "string", "description": 'The container, e.g. "bowl" or "pan_1".'}, "wait": _WAIT_PARAM},
        ["id"],
    ),
    _tool("lift_out", "Lift the held tool back out of the container it is lowered into.", {"wait": _WAIT_PARAM}),
    _tool(
        "squeeze",
        "Squeeze the held tool's trigger or handles and keep squeezing: an egg cracker cracks the "
        "egg inside it into whatever it is over; an electric whisk runs for as long as you squeeze. "
        "This does not let go of the tool (that is place). Lower the tool into its container first.",
        {"wait": _WAIT_PARAM},
    ),
    _tool(
        "release",
        "Stop squeezing the held tool (an electric whisk stops). You keep holding it.",
        {"wait": _WAIT_PARAM},
    ),
    _tool(
        "stir",
        "Stir with the held spoon or spatula, lowered into a bowl or pan first, for `seconds`. "
        "In a bowl it mixes (cracked eggs get beaten, more slowly than with a whisk); in a pan "
        "on a lit burner it scrambles eggs as they cook. The robot stirs the whole time; an "
        "event can wake you early (the stirring then goes on until the time is up or you stop).",
        {"seconds": {"type": "number", "description": "How long to stir, up to 300."}, "wait": _WAIT_PARAM},
        ["seconds"],
    ),
    _tool(
        "pour",
        "Pour what is in the held container (a bowl, a pan) into a visible container or onto a "
        "plate. Align to the target first. Fails with not_holding, unknown_id, not_visible, "
        "unreachable or no_placement (it does not go there).",
        {"id": {"type": "string", "description": 'Where it goes, e.g. "pan_1" or "plate".'}, "wait": _WAIT_PARAM},
        ["id"],
    ),
]

# The robot capabilities each tool needs (interface.robot_api); tools not listed need none.
TOOL_NEEDS: dict[str, frozenset[str]] = {
    "go_to": frozenset({BASE}), "align": frozenset({BASE}),
    "pick": frozenset({GRASP}), "place": frozenset({GRASP}), "holding": frozenset({GRASP}),
    "open": frozenset({ARTICULATE}), "close": frozenset({ARTICULATE}),
    "turn_on": frozenset({APPLIANCES}), "turn_off": frozenset({APPLIANCES}),
    "lower_into": frozenset({HANDHELD_TOOLS}), "lift_out": frozenset({HANDHELD_TOOLS}),
    "squeeze": frozenset({TRIGGER_TOOLS}), "release": frozenset({TRIGGER_TOOLS}),
    "stir": frozenset({HANDHELD_TOOLS}),
    "pour": frozenset({POUR}),
}  # fmt: skip

_STEP_ID = {"type": "string", "description": 'A step id from the plan, e.g. "s3".'}

# The agent loop's tools (AGENT_SPEC §7), added with ToolRegistry.add_loop_tools.
LOOP_SCHEMAS: list[dict] = [
    _tool(
        "wait",
        "Let time pass. Returns on the first of: the current motion ends, an event arrives (a "
        "timer is up, a watched step's check passed or timed out, a safety event), or timeout_s "
        "seconds pass. Use it instead of polling while food cooks or a timer runs. Fails with "
        "nothing_to_wait_for if nothing is moving, no timer or monitor is running and no "
        "timeout_s is given.",
        {"timeout_s": {"type": "number", "description": "Seconds to wait at most."}},
    ),
    _tool(
        "complete",
        "Mark a step done. Checks the step's done_when against the kitchen first: on a pass the "
        "step is done (your last note on it becomes its log line); on a fail it stays as it is "
        "and you get check_failed with the reason. The only way to mark a step done.",
        {"step_id": _STEP_ID},
        ["step_id"],
    ),
    _tool(
        "set_status",
        "Change a step's status: active (working on it now; one step at a time, and its "
        "depends_on must be done), waiting (progressing without you, e.g. cooking), pausing "
        "(winding it down to do something else), paused (stopped partway: note says what is "
        "left and where things are), failed (note says why). Use complete for done and "
        "skip_step for skipped.",
        {
            "step_id": _STEP_ID,
            "status": {"type": "string", "enum": ["active", "waiting", "pausing", "paused", "failed"]},
            "note": {"type": "string", "description": "Resume note (paused) or reason (failed); optional otherwise."},
        },
        ["step_id", "status"],
    ),
    _tool(
        "add_step",
        "Add a step the plan is missing (e.g. clear a burner that is in use). Say why.",
        {
            "goal": {"type": "string"},
            "done_when": {"type": "string", "description": "What must be true in the kitchen for it to be done."},
            "reason": {"type": "string"},
            "after": {"type": "string", "description": "Insert after this step id (default: at the end)."},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        ["goal", "done_when", "reason"],
    ),
    _tool(
        "skip_step",
        "Skip a step that is not needed, with the reason.",
        {"step_id": _STEP_ID, "reason": {"type": "string"}},
        ["step_id", "reason"],
    ),
    _tool(
        "request_replan",
        "Ask for a new plan when this one cannot work (an ingredient is missing, the dish has to "
        "change). Do not improvise a different dish.",
        {"reason": {"type": "string"}},
        ["reason"],
    ),
    _tool(
        "finish",
        "End the run with a summary. Allowed once every step is done, skipped or failed; "
        "abandon=true ends it anyway.",
        {"summary": {"type": "string"}, "abandon": {"type": "boolean"}},
        ["summary"],
    ),
    _tool(
        "set_timer",
        "Start a timer that sends an event after `seconds`. Setting a running timer again restarts it.",
        {
            "name": {"type": "string", "description": 'e.g. "egg"'},
            "seconds": {"type": "number"},
            "step_id": {"type": "string", "description": "The step it belongs to (shown on the plan)."},
        },
        ["name", "seconds"],
    ),
    _tool("cancel_timer", "Cancel a running timer.", {"name": {"type": "string"}}, ["name"]),
    _tool(
        "watch",
        "Check a step's done_when every every_s seconds until it holds (event: monitor passed; "
        "then call complete) or timeout_s passes (event: monitor timed out). For things that "
        "finish on their own, like cooking.",
        {"step_id": _STEP_ID, "every_s": {"type": "number"}, "timeout_s": {"type": "number"}},
        ["step_id", "every_s", "timeout_s"],
    ),
]


class ToolRegistry:
    """The tools the agent gets: those the robot's capabilities allow, minus `exclude`,
    and only those in `only` when it is given (task files and agent.yaml set these)."""

    def __init__(
        self,
        robot: RobotAPI,
        detector: Detector,
        world_map: WorldMap,
        config: AgentConfig,
        exclude: tuple[str, ...] = (),
        only: tuple[str, ...] | None = None,
    ):
        self.tools = NavTools(robot, detector, world_map, config)
        self.manip = ManipTools(robot, self.tools)
        self.kitchen = KitchenTools(robot, self.tools)
        self.handheld = HandheldTools(robot, self.tools)
        self._capabilities = frozenset(robot.capabilities())
        self._exclude, self._only = set(exclude), (set(only) if only is not None else None)
        self._schemas: dict[str, dict] = {}
        self._handlers = {}
        for schema in TOOL_SCHEMAS + HANDHELD_SCHEMAS:
            name = schema["function"]["name"]
            if self.allowed(name):
                owner = next(o for o in (self.manip, self.kitchen, self.handheld, self.tools) if hasattr(o, name))
                self._add(schema, getattr(owner, name))

    def allowed(self, name: str) -> bool:
        if name in self._exclude or (self._only is not None and name not in self._only):
            return False
        return TOOL_NEEDS.get(name, frozenset()) <= self._capabilities

    def has(self, name: str) -> bool:
        return name in self._handlers

    @property
    def schemas(self) -> list[dict]:
        return list(self._schemas.values())

    def add(self, owner, schemas: list[dict]) -> None:
        """Register `owner`'s methods under these schemas; a name already
        registered gets the new schema (and the new handler if `owner` has it)."""
        for schema in schemas:
            name = schema["function"]["name"]
            handler = getattr(owner, name, None) or self._handlers.get(name)
            if handler is None:
                raise ValueError(f"no handler for tool {name}")
            self._add(schema, handler)

    def add_loop_tools(self, plan_tools, time_tools) -> None:
        """The agent loop's tools; `wait` keeps its handler but wakes on events."""
        self.tools.time = time_tools
        time_tools.on_motion_end = self.tools._finish
        for schema in LOOP_SCHEMAS:
            name = schema["function"]["name"]
            if not self.allowed(name):
                continue
            owner = next((o for o in (plan_tools, time_tools) if hasattr(o, name)), None)
            handler = self._handlers.get(name) if name == "wait" else getattr(owner, name)
            self._add(schema, handler)

    def _add(self, schema: dict, handler) -> None:
        name = schema["function"]["name"]
        self._schemas[name] = schema
        self._handlers[name] = handler

    def dispatch(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                False, "unknown_tool", f'No tool named "{name}". Tools: {", ".join(self._handlers)}.'
            )
        # A null optional argument means "use the default".
        args = {k: v for k, v in (args or {}).items() if v is not None}
        problem = _check_args(self._schemas[name], args)
        if problem is None:
            try:
                inspect.signature(handler).bind(**args)
            except TypeError as exc:
                problem = str(exc)
        if problem is not None:
            return ToolResult(False, "invalid_arguments", f"{name}: {problem}")
        return handler(**args)


_JSON_TYPES = {"string": str, "boolean": bool, "number": (int, float), "array": list}


def _check_args(tool: dict, args: dict[str, Any]) -> str | None:
    """Validate arguments against the tool's schema. Returns a problem or None."""
    schema = tool["function"]["parameters"]
    missing = [key for key in schema.get("required", []) if key not in args]
    if missing:
        return f"missing required argument {missing[0]!r}"
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            return f'unexpected argument "{key}"'
        expected = _JSON_TYPES[spec["type"]]
        if not isinstance(value, expected) or (spec["type"] == "number" and isinstance(value, bool)):
            return f'"{key}" must be a {spec["type"]}'
        if spec["type"] == "array" and not all(isinstance(v, str) for v in value):
            return f'"{key}" must be an array of strings'
        if "enum" in spec and value not in spec["enum"]:
            return f'"{key}" must be one of {", ".join(spec["enum"])}'
    return None
