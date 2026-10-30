"""Task types: an outcome to achieve in a setup, defined once and used in any of its scenes.

A task type says how an instance starts (reset, start pose), what is measured every control step, and how the end
is scored. It works through the scene (autoresearch/scene.py), and it runs on the harness side: a skill never sees
it. A setup's task entry binds a type's arguments to that scene's objects and may override its defaults:

    pick: {type: pick_up, object: banana}

(agentic-skills-exploration's tasks/base.py.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

TYPES: dict[str, type["TaskType"]] = {}


def register(cls):
    TYPES[cls.name] = cls
    return cls


def get(name: str) -> "TaskType":
    import chefbot.autoresearch.tasks.pick_up  # noqa: F401  (registers the built-in types)

    if name not in TYPES:
        raise ValueError(f"no task type {name!r}; types: {', '.join(sorted(TYPES))}")
    return TYPES[name]()


@dataclass
class Task:
    """A task instance: a type bound to one setup's objects (a named entry in the setup)."""

    name: str
    type: "TaskType"
    args: dict
    start: dict = field(default_factory=dict)  # start-rule overrides


class TaskType:
    name: str = ""

    def bind(self, name: str, entry: dict) -> Task:
        """A setup's task entry -> a Task (arguments parsed, defaults filled)."""
        raise NotImplementedError

    def instruction(self, task: Task) -> str:
        """What an agent would be told."""
        raise NotImplementedError

    def reset_at(self, scene, task: Task, pose, seed: int) -> None:
        """The world's starting state for this task, for the robot starting at `pose`."""

    def clear(self, scene, task: Task) -> bool:
        """After the reset at a start pose: is the task's own setup sound there?"""
        return True

    def skill_args(self, task: Task) -> dict:
        """The arguments a skill gets for this task instance (pick_up: the object's name); {} for none."""
        return {}

    def start_poses(self, scene, task: Task, robot, seed: int):
        """Candidate start poses (world Pose2D), in order; the host uses the first where the robot is clear."""
        raise NotImplementedError

    def measure(self, scene, task: Task) -> dict:
        """Every control step: privileged measurements (ground truth), for scoring and the researcher's trace."""
        return {}

    def score(self, history: list[dict], task: Task) -> dict:
        """At the end, from the measurements: {"success": ..., ...}."""
        raise NotImplementedError
