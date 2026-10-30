"""pick_up: pick up a named object from a kitchen counter in front of the robot, among two others, arm only.

Start rule: the robot stands facing a counter, its base center 0.50-0.62 m out from the counter's front edge, yaw
jittered, at a place along the counter drawn from the seed. The target stands on the counter in front of it, 10-16 cm
in from the edge, turned at random; the two others beside it, 14-20 cm to either side (further for long objects, so
that they never overlap however they are turned). A start is clear when the
three stand flat on the counter's own top (no appliance, sink or edge under them) and nothing touches the robot.
Success: at the end the target is held between the fingertip pads and its bottom is at least LIFT_M above the
counter. The skill is told what to pick up: skill_args -> {"object": <the target's category, in words>}.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.autoresearch.tasks import Task, TaskType, register
from chefbot.interface.types import Pose2D

LIFT_M = 0.05
OUT_M = (0.50, 0.62)  # base center from the counter's front edge
END_MARGIN_M = 0.35  # the target this far from either end of the counter, at least
MAX_STARTS = 120  # starts tried before giving up (a kitchen with corners turns many down: the base meets the other run)


@register
class PickUp(TaskType):
    name = "pick_up"

    def bind(self, name: str, entry: dict) -> Task:
        return Task(name, self, {"object": entry["object"], "lift_m": float(entry.get("lift_m", LIFT_M))}, dict(entry.get("start") or {}))

    def instruction(self, task: Task) -> str:
        return f"Pick up the {task.args.get('words', 'object')}."

    def skill_args(self, task: Task) -> dict:
        return {"object": task.args["words"]} if "words" in task.args else {}

    def start_poses(self, scene, task: Task, robot, seed: int):
        rng = np.random.default_rng([seed, 3])
        counters = [c for c in scene.counters() if np.linalg.norm(c.box.width) >= 2 * END_MARGIN_M]
        tried = 0
        for k in rng.permutation(len(counters)):
            c = counters[int(k)]
            width = float(np.linalg.norm(c.box.width))
            spots = np.arange(END_MARGIN_M, width - END_MARGIN_M + 1e-9, 0.1)
            for along in rng.permutation(spots):
                if tried >= MAX_STARTS:
                    return
                tried += 1
                u = float(along) / width
                front = c.box.point(u, 0.0, 1.0)[:2]
                facing = c.box.facing
                p = front + rng.uniform(*OUT_M) * facing
                yaw = math.atan2(-facing[1], -facing[0]) + math.radians(rng.uniform(-10, 10))
                task.args["spot"] = (c.name, u)
                yield Pose2D(float(p[0]), float(p[1]), float(yaw))

    def reset_at(self, scene, task: Task, pose, seed: int) -> None:
        rng = np.random.default_rng([seed, 5])
        counter = scene.fixture(task.args["spot"][0])
        box = counter.box
        width, depth = float(np.linalg.norm(box.width)), float(np.linalg.norm(box.depth))
        target = scene.object(task.args["object"])
        others = [scene.object(n) for n in scene.objects() if n != task.args["object"]]
        task.args["words"] = target.category.replace("_", " ")
        for i, n in enumerate(scene.objects()):
            scene.object(n).park(i)
        scene.forward()
        inset = float(rng.uniform(0.10, 0.16))
        plan = [(target, 0.0, inset)]
        sides = rng.permutation([-1.0, 1.0])
        gap = [max(rng.uniform(0.14, 0.20), target.reach + o.reach + 0.02) for o in others[:2]]
        plan += [(o, sides[k] * gap[k], inset + rng.uniform(-0.03, 0.03)) for k, o in enumerate(others[:2])]
        task.args["placed"] = False
        for obj, du, v in plan:
            u = task.args["spot"][1] + du / width
            v = max(v, obj.radius + 0.03) / depth
            if not 0.0 < u < 1.0 or v >= 1.0:
                return
            x, y, _ = box.point(u, v, 1.0)
            z = scene.flat_height(counter, x, y, obj.radius)
            if z is None:
                return
            obj.place(float(x), float(y), z, float(rng.uniform(-math.pi, math.pi)))
            if obj is target:
                task.args["counter_z"] = z
        scene.forward()
        task.args["placed"] = True

    def clear(self, scene, task: Task) -> bool:
        """The three stand where they were put: on the counter, touching nothing but it, each other and the floor."""
        if not task.args.get("placed"):
            return False
        ours = set().union(*(scene.object(n).geoms for n in scene.objects()))
        return not scene.penetrating(ours, {scene.fixture(task.args["spot"][0]).root})

    def measure(self, scene, task: Task) -> dict:
        t = scene.object(task.args["object"])
        lo, _ = t.box()
        return {"lift_m": float(lo[2] - task.args.get("counter_z", 0.0)), "held": t.body in scene.env.grasped_bodies()}

    def score(self, history: list[dict], task: Task) -> dict:
        measured = [h for h in history if "lift_m" in h]
        if not measured:
            return {"success": None, "lift_cm": None, "held": None, "max_lift_cm": None}
        end = measured[-1]
        ok = bool(end["held"]) and end["lift_m"] >= task.args["lift_m"]
        return {"success": ok, "lift_cm": round(end["lift_m"] * 100, 1), "held": bool(end["held"]),
                "max_lift_cm": round(max(h["lift_m"] for h in measured) * 100, 1), "object": task.args.get("words")}  # fmt: skip
