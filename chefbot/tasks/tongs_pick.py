"""tongs_pick: pick up the tongs from the counter, then use them to lift a named food, on the counter or in a pan,
bowl or plate on it. MuJoCo kitchens (environments/mujoco_kitchen).

Start rule: the robot stands facing a counter, its base center 0.36-0.44 m out from the front edge, yaw jittered.
The food stands in front of it, a little to one side: on the counter, or in its container (a pan, bowl or plate)
there. The tongs lie on the counter to the other side, clear of the food's holder (turned up to 30 degrees from along
the edge, either end first). A distractor food stands beside the food, away from the tongs. A start is clear when everything stands flat where it was put and nothing touches the robot.

Fragile foods break: each episode draws the force that breaks each fragile food in the scene (lognormal about its
category's median), and a food whose contact force (averaged over 10 ms) passes it breaks, visibly. A skill is told
only that foods can break. The thresholds are the harness's alone: never in a trace, a score or a skill's view.

Success, at the end: the food's bottom is at least LIFT_M above where it started, it touches the tongs and nothing of
the robot, the tongs are in the hand (touching the fingers) and rest on nothing else, the food is not broken, and
the hand never touched the food. skill_args -> {"object": the food in words, "where": "on the counter" / "in the
bowl" ..., "tool": "tongs"}.
"""

from __future__ import annotations

import math

import numpy as np

from chefbot.interface.types import Pose2D
from chefbot.tasks import Task, TaskType, register

LIFT_M = 0.05
OUT_M = (0.36, 0.44)
END_MARGIN_M = 0.45
MAX_STARTS = 120
FORCE_WINDOW = 5  # physics steps (10 ms) the contact force is averaged over
# Breaking force, N: (median, log-sd). Measured crush forces of whole produce are higher; a tong's two narrow
# tips concentrate it (eggs crack at ~25-40 N between flat plates).
FRAGILE = {
    "egg": (22.0, 0.25),
    "strawberry": (6.0, 0.30),
    "tomato": (24.0, 0.30),
    "peach": (18.0, 0.30),
    "apricot": (15.0, 0.30),
    "kiwi": (20.0, 0.30),
    "banana": (20.0, 0.30),
    "donut": (8.0, 0.30),
    "croissant": (5.0, 0.30),
}
BROKEN_RGBA = {"egg": (0.95, 0.78, 0.25, 1.0)}  # yolk; others go dark and dull
CONTAINER_WORDS = {"pan": "pan", "bowl": "bowl", "plate": "plate"}


@register
class TongsPick(TaskType):
    name = "tongs_pick"

    def bind(self, name: str, entry: dict) -> Task:
        args = {"object": entry["object"], "container": entry.get("container"), "distractors": list(entry.get("distractors") or []),
                "tongs": entry.get("tongs", "tongs"), "lift_m": float(entry.get("lift_m", LIFT_M))}  # fmt: skip
        return Task(name, self, args, dict(entry.get("start") or {}))

    def instruction(self, task: Task) -> str:
        return f"Pick up the tongs and use them to lift the {task.args.get('words', 'food')} {task.args.get('where', '')}.".replace(" .", ".")

    def skill_args(self, task: Task) -> dict:
        return {"object": task.args["words"], "where": task.args["where"], "tool": "tongs"} if "words" in task.args else {}

    # -- start -------------------------------------------------------------------------

    def start_poses(self, scene, task: Task, robot, seed: int):
        rng = np.random.default_rng([seed, 3])
        counters = [c for c in scene.counters() if np.linalg.norm(c.box.width) >= 2 * END_MARGIN_M]
        tried = 0
        for k in rng.permutation(len(counters)):
            c = counters[int(k)]
            width = float(np.linalg.norm(c.box.width))
            for along in rng.permutation(np.arange(END_MARGIN_M, width - END_MARGIN_M + 1e-9, 0.1)):
                if tried >= MAX_STARTS:
                    return
                tried += 1
                u = float(along) / width
                front = c.box.point(u, 0.0, 1.0)[:2]
                facing = c.box.facing
                p = front + rng.uniform(*OUT_M) * facing
                yaw = math.atan2(-facing[1], -facing[0]) + math.radians(rng.uniform(-8, 8))
                task.args["spot"] = (c.name, u)
                yield Pose2D(float(p[0]), float(p[1]), float(yaw))

    def reset_at(self, scene, task: Task, pose, seed: int) -> None:
        rng = np.random.default_rng([seed, 5])
        a = task.args
        counter = scene.fixture(a["spot"][0])
        box = counter.box
        width, depth = float(np.linalg.norm(box.width)), float(np.linalg.norm(box.depth))
        along = box.width / width
        target = scene.object(a["object"])
        container = scene.object(a["container"]) if a.get("container") else None
        tongs = scene.extra(a["tongs"])
        a["words"] = target.category.replace("_", " ")
        a["where"] = f"in the {CONTAINER_WORDS.get(container.category, container.category)}" if container else "on the counter"
        for i, n in enumerate(scene.objects()):
            scene.object(n).park(i)
        tongs.park(len(scene.objects()))
        scene.forward()
        a["placed"] = False
        side = float(rng.choice([-1.0, 1.0]))  # the tongs' side; the food is a little to the other

        def spot(du: float, inset: float, radius: float):
            u = a["spot"][1] + du / width
            v = max(inset, radius + 0.03) / depth
            if not 0.0 < u < 1.0 or v >= 0.9:
                return None
            x, y, _ = box.point(u, v, 1.0)
            z = scene.flat_height(counter, float(x), float(y), radius)
            return None if z is None else (float(x), float(y), z)

        # The food: on the counter, or in its container there, a little to one side of the robot.
        holder = container or target
        hs = spot(-side * rng.uniform(0.0, 0.06), rng.uniform(0.10, 0.15), holder.radius)
        if hs is None:
            return
        if container:
            container.place(*hs, float(rng.uniform(-math.pi, math.pi)))
            scene.forward()
            dx, dy = rng.uniform(-0.25, 0.25, 2) * max(container.radius - target.reach, 0.0)
            hit, z = scene.ray_down(hs[0] + dx, hs[1] + dy, hs[2] + 0.4)
            if hit != container.body:
                return
            target.place(hs[0] + dx, hs[1] + dy, z, float(rng.uniform(-math.pi, math.pi)))
        else:
            target.place(*hs, float(rng.uniform(-math.pi, math.pi)))
        scene.forward()
        # The tongs: lying on the counter on the other side, clear of the food's holder, along the edge give or take
        # 30 degrees, either end first.
        edge_yaw = math.atan2(along[1], along[0])
        turn = math.radians(rng.uniform(-30, 30))
        t_yaw = edge_yaw + turn + (math.pi if rng.random() < 0.5 else 0.0)
        half_along = tongs.size[0] / 2 * abs(math.cos(turn)) + tongs.size[1] / 2 * abs(math.sin(turn))
        hu = float((np.asarray(hs[:2]) - box.origin[:2]) @ along[:2]) - a["spot"][1] * width
        s = spot(hu + side * (holder.reach + half_along + rng.uniform(0.03, 0.08)), rng.uniform(0.07, 0.12), 0.03)
        if s is None or not self._lies_flat(scene, counter, tongs, s, t_yaw):
            return
        tongs.place(s[0], s[1], s[2], t_yaw)
        # A distractor beside the food, on its other side.
        gap = holder.reach + 0.03
        for k, name in enumerate(a["distractors"][:2]):
            o = scene.object(name)
            ds = spot(-side * (rng.uniform(0.02, 0.10) + gap + o.reach + 0.02 + 0.12 * k), rng.uniform(0.08, 0.16), o.radius)
            if ds is not None:
                o.place(*ds, float(rng.uniform(-math.pi, math.pi)))
            gap += o.reach * 2 + 0.03
        scene.forward()
        a["placed"] = True
        self._arm_fragility(scene, task, seed)

    def _lies_flat(self, scene, counter, tongs, s, yaw) -> bool:
        """The counter is flat and free under the tongs' whole length."""
        half = tongs.size[0] / 2
        for f in (-1.0, -0.5, 0.5, 1.0):
            x, y = s[0] + f * half * math.cos(yaw), s[1] + f * half * math.sin(yaw)
            if scene.flat_height(counter, x, y, 0.02) is None:
                return False
        return True

    def clear(self, scene, task: Task) -> bool:
        a = task.args
        if not a.get("placed"):
            return False
        names = [a["object"], *a["distractors"][:2]] + ([a["container"]] if a.get("container") else [])
        ours = set().union(*(scene.object(n).geoms for n in names), scene.extra(a["tongs"]).geoms)
        counter = scene.env.body_root(scene.fixture(a["spot"][0]).root)
        if scene.penetrating(ours, {counter}):
            return False
        # The food settled where it was put (in its container, or on the counter), not on the floor; lift is
        # measured from where it settled.
        target = scene.object(a["object"])
        bottom = float(target.box()[0][2])
        a["start_bottom"] = bottom
        return bottom > scene.fixture(a["spot"][0]).box.top - 0.02 and float(scene.extra(a["tongs"]).box()[0][2]) > scene.fixture(a["spot"][0]).box.top - 0.02

    # -- fragile food (harness side) -------------------------------------------------------

    def _arm_fragility(self, scene, task: Task, seed: int) -> None:
        env = scene.env
        rng = np.random.default_rng([seed, 11])
        foods = {}
        for n in scene.objects():
            o = scene.object(n)
            if o.category in FRAGILE:
                median, sd = FRAGILE[o.category]
                foods[o.body] = {"name": n, "limit": float(median * math.exp(sd * rng.standard_normal())), "recent": [], "broken": False}
        state = {"foods": foods, "events": []}
        task.args["_fragile"] = state  # private: never reported (score() and measure() give only "broken")
        m = env.model

        def watch(e):
            if not foods:
                return
            total = dict.fromkeys(foods, 0.0)
            for g1, g2, f in e.contacts():
                for g in (g1, g2):
                    r = int(m.body_rootid[m.geom_bodyid[g]])
                    if r in total:
                        total[r] += f
            for r, info in foods.items():
                info["recent"].append(total[r])
                del info["recent"][:-FORCE_WINDOW]
                if not info["broken"] and len(info["recent"]) == FORCE_WINDOW and np.mean(info["recent"]) > info["limit"]:
                    info["broken"] = True
                    state["events"].append({"name": info["name"], "t": round(float(e.data.time), 2)})
                    self._show_broken(scene, r, scene.object(info["name"]).category)

        env.physics_hooks.append(watch)

    def _show_broken(self, scene, root: int, category: str) -> None:
        m = scene.env.model
        rgba = BROKEN_RGBA.get(category)
        for g in np.flatnonzero(m.body_rootid[m.geom_bodyid] == root):
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                continue
            m.geom_matid[g] = -1
            m.geom_rgba[g] = rgba if rgba else (0.25, 0.15, 0.1, 1.0)
            if rgba is None:
                m.geom_size[g] = m.geom_size[g]  # same shape, crushed colour

    # -- measure and score ---------------------------------------------------------------

    def measure(self, scene, task: Task) -> dict:
        a = task.args
        env = scene.env
        m = env.model
        target = scene.object(a["object"])
        tongs = scene.extra(a["tongs"])
        touch = {"target_tongs": False, "target_robot": False, "tongs_fingers": False, "tongs_other": False}
        for g1, g2 in env._touching():
            r1, r2 = int(m.body_rootid[m.geom_bodyid[g1]]), int(m.body_rootid[m.geom_bodyid[g2]])
            for (ga, ra), (gb, rb) in (((g1, r1), (g2, r2)), ((g2, r2), (g1, r1))):
                if ra == target.body:
                    if rb == tongs.body:
                        touch["target_tongs"] = True
                    elif rb == env.robot_root:
                        touch["target_robot"] = True
                if ra == tongs.body:
                    if gb in env.finger_geoms:
                        touch["tongs_fingers"] = True
                    elif rb not in (env.robot_root, target.body):
                        touch["tongs_other"] = True
        frag = a.get("_fragile", {}).get("foods", {})
        broken = bool(frag.get(target.body, {}).get("broken", False))
        others = sorted(i["name"] for r, i in frag.items() if i["broken"] and r != target.body)
        return {"lift_m": float(target.box()[0][2] - a.get("start_bottom", 0.0)), **touch, "broken": broken, "others_broken": others}

    def score(self, history: list[dict], task: Task) -> dict:
        measured = [h for h in history if "lift_m" in h]
        if not measured:
            return {"success": None, "lift_cm": None, "broken": None, "tongs_held": None, "hand_touched_food": None}
        end = measured[-1]
        hand_touched = any(h["target_robot"] for h in measured)
        held = bool(end["tongs_fingers"]) and not end["tongs_other"]
        ok = (end["lift_m"] >= task.args["lift_m"] and end["target_tongs"] and not end["target_robot"] and held
              and not end["broken"] and not hand_touched)  # fmt: skip
        return {"success": bool(ok), "lift_cm": round(end["lift_m"] * 100, 1), "max_lift_cm": round(max(h["lift_m"] for h in measured) * 100, 1),
                "broken": bool(end["broken"]), "tongs_held": held, "food_in_tongs": bool(end["target_tongs"]),
                "hand_touched_food": hand_touched, "others_broken": end["others_broken"], "object": task.args.get("words"),
                "where": task.args.get("where")}  # fmt: skip
