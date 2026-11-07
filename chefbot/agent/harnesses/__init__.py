"""Harnesses: the loops an agent runs in, by name (an agent's agent.yaml `harness:`). A harness module exposes
AgentSession (AgentSession.build wires a run), AgentRunner (runs it to the end) and INTERFACES (the robot APIs it
runs sessions with).

    harness = harnesses.get("plan_loop")
    session = harness.AgentSession.build(...)
"""

from __future__ import annotations

import importlib
from types import ModuleType

HARNESSES = {
    "plan_loop": "chefbot.agent.harnesses.plan_loop",  # one tool call per turn against a plan (AGENT_SPEC §8)
}


def names() -> list[str]:
    return sorted(HARNESSES)


def get(name: str) -> ModuleType:
    if name not in HARNESSES:
        raise ValueError(f"no harness {name!r}; harnesses: {', '.join(names())}")
    return importlib.import_module(HARNESSES[name])
