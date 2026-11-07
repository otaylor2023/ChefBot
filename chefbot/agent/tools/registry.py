"""Tool registry: function-calling schemas for the LLM, and dispatch.

The workflow the descriptions teach:
    go_to -> detect -> align -> pick_up -> go_to -> detect -> align -> place

Constructed with a RobotAPI and a Detector; it never knows which env is behind them. The tools that do
not move the robot (detect, wait, the plan's and time's) are defined in chefbot/tools; the ones that do
are skills (skills/<name>/robotapi_v1), run in this process.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import functools

from chefbot import skills
from chefbot import tools as toolbox
from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.manip import HandheldTools, KitchenTools, ManipTools
from chefbot.agent.tools.nav import NavTools
from chefbot.agent.world.map import WorldMap
from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.types import ToolResult


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


# The tools that move the robot are skills (skills/<name>/robotapi_v1, robot-api-v1: run(ctx, **args) in this
# process); the rest are chefbot/tools'. In the order the LLM sees them.
ROBOT_SKILLS = ("go_to", "align", "pick_up", "place", "open", "close", "turn_on", "turn_off",
                "lower_into", "lift_out", "squeeze", "release", "stir", "pour")  # fmt: skip
SKILLS = {name: skills.load(f"{name}/robotapi_v1") for name in ROBOT_SKILLS}

TOOL_SCHEMAS: list[dict] = [
    SKILLS["go_to"].schema,
    toolbox.schema("detect"),
    toolbox.schema("nearby"),
    SKILLS["align"].schema,
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
    toolbox.schema("get_status"),
    toolbox.schema("stop"),
    toolbox.schema("get_pose"),
    *(SKILLS[name].schema for name in ("pick_up", "place", "open", "close", "turn_on", "turn_off")),
    toolbox.schema("holding"),
]

# Handheld tools and pouring (offered when the robot has HANDHELD_TOOLS / POUR).
HANDHELD_SCHEMAS: list[dict] = [SKILLS[name].schema for name in ("lower_into", "lift_out", "squeeze", "release", "stir", "pour")]


# The agent loop's tools (AGENT_SPEC §7, defined in chefbot/tools), added with ToolRegistry.add_loop_tools.
# `wait` here is the loop's: it replaces the motion-only one above and also wakes on events.
LOOP_TOOLS = ("wait", "complete", "set_status", "add_step", "skip_step", "request_replan", "finish",
              "set_timer", "cancel_timer", "watch")  # fmt: skip
LOOP_SCHEMAS: list[dict] = [toolbox.schema(name) for name in LOOP_TOOLS]


@dataclass(frozen=True)
class SkillContext:
    """What a robot-api-v1 skill's run(ctx, ...) gets: the session's RobotAPI and the agent's robot-side tools,
    which share the detections, the map and the held object."""

    robot: RobotAPI
    nav: NavTools
    manip: ManipTools
    kitchen: KitchenTools
    handheld: HandheldTools


class ToolRegistry:
    """The tools the agent gets: those the robot's capabilities allow, minus `exclude`,
    and only those in `only` when it is given (task files and agent.yaml set these). Of chefbot/tools' tools,
    only those in `agent_tools` when it is given (the agent's agent.yaml `tools:`). Of the robot-api-v1 skills, those in
    `robot_skills` when it is given (the session's), else the 14 above; one a slot above does not name comes last."""

    def __init__(
        self,
        robot: RobotAPI,
        detector: Detector,
        world_map: WorldMap,
        config: AgentConfig,
        exclude: tuple[str, ...] = (),
        only: tuple[str, ...] | None = None,
        agent_tools: tuple[str, ...] | None = None,
        robot_skills: list[skills.Skill] | None = None,
    ):
        self.tools = NavTools(robot, detector, world_map, config)
        self.manip = ManipTools(robot, self.tools)
        self.kitchen = KitchenTools(robot, self.tools)
        self.handheld = HandheldTools(robot, self.tools)
        self._capabilities = frozenset(robot.capabilities())
        self._exclude, self._only = set(exclude), (set(only) if only is not None else None)
        self._agent_tools = set(agent_tools) if agent_tools is not None else None
        self._owners = {"nav": self.tools, "manip": self.manip}
        self._ctx = SkillContext(robot, self.tools, self.manip, self.kitchen, self.handheld)
        given = list(SKILLS.values()) if robot_skills is None else list(robot_skills)
        not_here = [s.path for s in given if s.interface != "robot-api-v1"]
        if not_here:
            raise ValueError(f"skills {not_here} are not robot-api-v1: they do not run in the agent's process")
        self._skills = {s.name: s for s in given}
        self._schemas: dict[str, dict] = {}
        self._handlers = {}
        for schema in TOOL_SCHEMAS + HANDHELD_SCHEMAS + [s.schema for s in given if s.name not in SKILLS]:
            name = schema["function"]["name"]
            if name not in toolbox.names():  # a skill's slot
                if name not in self._skills:
                    continue  # the session has no skill by this name
                schema = self._skills[name].schema
            if self.allowed(name):
                self._add(schema, self._handler(name))

    def allowed(self, name: str) -> bool:
        if name in self._exclude or (self._only is not None and name not in self._only):
            return False
        registered = name in toolbox.names()
        if registered and self._agent_tools is not None and name not in self._agent_tools:
            return False
        needs = toolbox.get(name).needs if registered else self._skills[name].needs
        return needs <= self._capabilities

    def _handler(self, name: str):
        """A chefbot/tools tool's handler is its entry; a skill's is its run(ctx, **args), ctx the robot-side owners."""
        if name in toolbox.names():
            owner, method = toolbox.get(name).entry
            return getattr(self._owners[owner], method)
        return functools.partial(self._skills[name].function(), self._ctx)

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

    def add_tool(self, schema: dict, handler) -> None:
        """A tool from outside the agent's process: a chefbot-v1 skill, its handler one sandboxed episode
        (runtime/agent_skills.py). Its arguments are checked against the schema like any other's."""
        self._add(schema, handler)

    def add_loop_tools(self, plan_tools, time_tools) -> None:
        """The agent loop's tools; `wait` keeps its handler but wakes on events."""
        self.tools.time = time_tools
        time_tools.on_motion_end = self.tools._finish
        self._owners.update(plan=plan_tools, time=time_tools)
        for schema in LOOP_SCHEMAS:
            name = schema["function"]["name"]
            if self.allowed(name):
                self._add(schema, self._handler(name))

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
