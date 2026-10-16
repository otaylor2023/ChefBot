"""Ground truth of a RoboCasa kitchen as relations, a cook model, safety, and run time.

  RoboCasaKitchen   who holds what, what is on / in what, doors and burners,
                    how cooked the food is; safety (an empty lit burner is turned
                    off, burnt food is reported)
  RoboCasaClock     run time = sim time + skipped idle time: waiting while the
                    robot is idle skips ahead without stepping physics (nothing is
                    moving; cooking is time-based)
  RoboCasaObserver  the relations as text, for the step checker and the run log

Same text format as the symbolic kitchen, so the judge, the success rules and
the run log work unchanged.
"""

from __future__ import annotations

import numpy as np

from chefbot.envs.robocasa.appliances import burner_on, burner_position, burners, set_burner
from chefbot.envs.robocasa.detector import named_fixtures, scene_objects
from chefbot.envs.robocasa.env import TOOL_BIT, RoboCasaEnv
from chefbot.envs.robocasa.objects import object_points
from chefbot.envs.symbolic.world import Timing
from chefbot.interface.clock import Clock
from chefbot.interface.observer import Observation, Observer

COOKWARE = {"pan", "pot", "saucepan", "kettle"}
CONTAINERS = COOKWARE | {"bowl", "mug", "cup", "fridge", "microwave", "oven", "dishwasher", "sink", "toaster"}
FOOD = {"sausage", "hot dog", "steak", "tomato", "corn", "potato", "bacon", "egg", "bread", "pancake", "broccoli",
        "carrot", "onion", "mushroom", "fish", "chicken breast", "pork chop", "lamb chop"}  # fmt: skip
OPENABLE = {"fridge", "microwave", "oven", "dishwasher"}
UTENSILS = {"wooden spoon", "spoon", "spatula", "whisk", "ladle"}  # stirred with, held
SPOILED = {"burnt", "overcooked"}
_UPDATE_EVERY = 0.5  # s of run time between cook / safety updates
_ON_BURNER = 0.12  # m (xy) from a burner's site: on it
_STACK_MARGIN = 0.01


class RoboCasaKitchen:
    def __init__(self, env: RoboCasaEnv, actuator, timing: Timing | None = None):
        self.env = env
        self._actuator = actuator
        self.timing = timing or Timing.load()
        self.skipped = 0.0  # idle run time skipped without physics
        self.heat: dict[str, float] = {}
        self.stirred_hot: dict[str, float] = {}  # stirred while being heated (tossing, scrambling)
        self.stage: dict[str, str] = {}
        self.events: list[str] = []
        self._reported: set[str] = set()
        self._burner_since: dict[str, float] = {}
        self._empty_since: dict[str, float] = {}
        self._last = None
        self._rng = np.random.RandomState(1)
        with env.lock:
            self._burners = burners(env)
            self._fixtures = named_fixtures(env)
            objects = sorted(scene_objects(env))
        labels = [label for _, label, _ in objects]
        self.objects = []  # (name, label, body): names as the map's (label, numbered if repeated)
        seen: dict[str, int] = {}
        for sim, label, body in objects:
            seen[label] = seen.get(label, 0) + 1
            name = label if labels.count(label) == 1 else f"{label}_{seen[label]}"
            self.objects.append((name.replace(" ", "_"), label, body))
        # Utensils and cookware have an adaptive handle: a two-finger pinch on a
        # bare spoon handle twists and slips at the first push on the food, and a
        # pan swings nose-down about the pads under its own weight, so the gripper
        # locks onto them once closed (env.lockable). Cookware keeps its contacts.
        with env.lock:
            roots = {label: set() for label in UTENSILS | COOKWARE}
            for _, label, body in self.objects:
                if label in roots:
                    roots[label].add(int(env.model.body_rootid[body]))
            env.lockable |= set().union(*roots.values())
            env.keeps_contacts |= set().union(*(roots[label] for label in COOKWARE))
            # A locked utensil touches only the food (env.TOOL_BIT).
            for _, label, body in self.objects:
                if label in FOOD:
                    for geom in np.flatnonzero(env.model.body_rootid[env.model.geom_bodyid] == env.model.body_rootid[body]):
                        if env.model.geom_contype[geom] or env.model.geom_conaffinity[geom]:
                            env.model.geom_conaffinity[geom] |= TOOL_BIT

    # -- time -------------------------------------------------------------------

    def now(self) -> float:
        return float(self.env.data.time) + self.skipped

    def update(self, force: bool = False) -> None:
        """Bring cooking and safety up to now (heat over the time since the last update)."""
        now = self.now()
        if self._last is None:
            self._last = now
        dt = now - self._last
        if dt < _UPDATE_EVERY and not force:
            return
        self._last = now
        with self.env.lock:
            where = self.relations()
            lit = {b.name for b in self._burners if burner_on(self.env, b)}
        for burner in self._burners:
            if burner.name in lit:
                self._burner_since.setdefault(burner.name, now - dt)
            else:
                self._burner_since.pop(burner.name, None)
        names = {name: label for name, label, _ in self.objects}
        stirring = self._stirred_container(where)
        for name, label in names.items():
            if stirring and label in FOOD and where.get(name, ("", ""))[1] == stirring and self._heated(name, where, lit):
                self.stirred_hot[name] = self.stirred_hot.get(name, 0.0) + dt
            if label in FOOD and self._heated(name, where, lit):
                self.heat[name] = self.heat.get(name, 0.0) + dt
                stage = self.timing.stage(label, self.heat[name])
                if stage != self.stage.get(name, "raw"):
                    self.stage[name] = stage
                    if stage in SPOILED and name not in self._reported:
                        self._reported.add(name)
                        self.events.append(f"the {label} is {stage}")
        holders = {h for _, h in where.values()}
        for burner in self._burners:
            if burner.name not in lit or burner.name in holders:
                self._empty_since.pop(burner.name, None)
                continue
            # From the first update that saw it empty: one update can span a long
            # skipped wait, and a pan nudged by a spoon can read off for an instant.
            since = self._empty_since.setdefault(burner.name, now)
            if now - since >= self.timing.empty_burner_s:
                with self.env.lock:
                    set_burner(self.env, burner, False, self._rng)
                self._empty_since.pop(burner.name, None)
                self.events.append(
                    f"{burner.name} was on with nothing on it for {self.timing.empty_burner_s:.0f} s; the robot turned it off"
                )

    def _stirred_container(self, where: dict) -> str | None:
        """The container a held utensil is moving in right now, if any."""
        env = self.env
        with env.lock:
            if float(np.abs(env.get_state().arm_qd).max()) < 0.05:
                return None  # the arm is still: not stirring
            held = [(n, label, body) for n, label, body in self.objects if where.get(n, ("",))[0] == "held"]
            utensil = next(((n, body) for n, label, body in held if label in UTENSILS), None)
            if utensil is None:
                return None
            # Any part of it down in the container (a spoon held across its middle
            # tilts in the grip: its lowest point can be outside, over the rim).
            spoon = object_points(env, utensil[1])
            for name, label, body in self.objects:
                if label not in CONTAINERS or where.get(name, ("",))[0] == "held":
                    continue
                pts = object_points(env, body)
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                # Down to the rim, or to the top of the food piled in it (in it, or on what is in it).
                top = hi[2]
                for other, _, b in self.objects:
                    if other == name or other == utensil[0]:
                        continue
                    o = object_points(env, b)
                    middle = (o.min(axis=0) + o.max(axis=0)) / 2
                    if np.all(middle[:2] >= lo[:2]) and np.all(middle[:2] <= hi[:2]) and o[:, 2].min() >= lo[2] - 0.01:
                        top = max(top, float(o[:, 2].max()))
                inside = np.all((spoon[:, :2] >= lo[:2]) & (spoon[:, :2] <= hi[:2]), axis=1)
                inside &= (spoon[:, 2] >= lo[2] - 0.01) & (spoon[:, 2] <= top + 0.03)
                if inside.any():
                    return name
        return None

    def _heated(self, name: str, where: dict, lit: set[str]) -> bool:
        relation = where.get(name)
        if relation is None:
            return False
        holder = relation[1]
        holder_label = next((label for n, label, _ in self.objects if n == holder), None)
        if holder_label in COOKWARE:
            under = where.get(holder)
            return under is not None and under[1] in lit
        return False

    def safety(self, now: float) -> list[tuple[str, dict]]:
        """EventBus safety monitor."""
        self.update()
        events, self.events = self.events, []
        return [(message, {"source": "robot"}) for message in events]

    # -- relations --------------------------------------------------------------

    def relations(self) -> dict[str, tuple[str, str]]:
        """object name -> ("held", "") | ("on" | "inside", holder name). Call with env.lock held."""
        env = self.env
        held = env.grasped_bodies()
        boxes = {}
        for name, label, body in self.objects:
            pts = object_points(env, body)
            if len(pts):
                boxes[name] = (label, int(env.model.body_rootid[body]), pts.min(axis=0), pts.max(axis=0))
        out: dict[str, tuple[str, str]] = {}
        for name, (label, root, lo, hi) in boxes.items():
            if root in held:
                out[name] = ("held", "")
                continue
            center = (lo[:2] + hi[:2]) / 2
            under = [
                (other, olabel) for other, (olabel, oroot, olo, ohi) in boxes.items()
                if other != name and oroot not in held
                and np.all(center >= olo[:2] - _STACK_MARGIN) and np.all(center <= ohi[:2] + _STACK_MARGIN)
                # its bottom no higher than ours (a sausage lies about as low as the pan's base),
                # its top above our bottom (we are in it or on it), and it is the bigger of the two
                and olo[2] <= lo[2] + 0.005 and ohi[2] > lo[2] - 0.03
                and np.prod(ohi[:2] - olo[:2]) > np.prod(hi[:2] - lo[:2])
            ]  # fmt: skip
            if under:
                # In a container wins (a tomato leaning on the corn in the pan is in the pan), then the highest.
                other, olabel = max(under, key=lambda o: (o[1] in CONTAINERS, boxes[o[0]][3][2]))
                out[name] = ("inside" if olabel in CONTAINERS else "on", other)
                continue
            burner = next((b for b in self._burners
                           if np.linalg.norm(burner_position(env, b)[:2] - center) <= _ON_BURNER
                           and abs(burner_position(env, b)[2] - lo[2]) <= 0.08), None)  # fmt: skip
            if burner is not None:
                out[name] = ("on", burner.name)
                continue
            fixture = next((f for f in self._fixtures if f.box.contains_xy(center[0], center[1])), None)
            if fixture is not None:
                out[name] = ("inside" if fixture.label in CONTAINERS else "on", fixture.name)
            else:
                out[name] = ("on", "the floor")
        return out

    def describe(self) -> str:
        self.update(force=True)
        env, now = self.env, self.now()
        with env.lock:
            where = self.relations()
            lit = {b.name for b in self._burners if burner_on(env, b)}
            doors = {}
            if self._actuator is not None:
                from chefbot.interface.types import Detection

                for f in self._fixtures:
                    if f.label in OPENABLE:
                        state = self._actuator.get_state(Detection(f.label, f.box.center))
                        if state in ("open", "closed"):
                            doors[f.name] = state
        held = [n for n, (rel, _) in where.items() if rel == "held"]
        lines = [f"held: {held[0] if held else 'nothing'}"]
        for name in sorted(where):
            relation, holder = where[name]
            lines.append(f"{name}: held" if relation == "held" else f"{name}: {relation} {holder}")
        for burner in self._burners:
            if burner.name in lit:
                lines.append(f"{burner.name}: on (for {_clock(now - self._burner_since.get(burner.name, now))})")
            else:
                lines.append(f"{burner.name}: off")
        lines += [f"{name}: {state}" for name, state in doors.items()]
        for name, label, _ in self.objects:
            if label in FOOD:
                stage = self.stage.get(name, "raw")
                lines.append(f"{name}: cook state = {stage} ({_clock(self.heat.get(name, 0.0))} of heat)")
                if self.heat.get(name, 0.0) > 0:
                    stirred = self.stirred_hot.get(name, 0.0)
                    texture = "stir-fried" if stirred >= self.timing.scrambled_after else "not stirred while cooking"
                    lines.append(f"{name}: {texture} ({_clock(stirred)} stirred while cooking)")
        return "\n".join(lines)


class RoboCasaClock(Clock):
    """Run time; waiting while nothing moves skips ahead (no physics needed)."""

    def __init__(self, kitchen: RoboCasaKitchen):
        self._kitchen = kitchen

    def now(self) -> float:
        self._kitchen.update()
        return self._kitchen.now()

    def sleep(self, seconds: float) -> None:
        # Called only while the robot is idle (TimeTools): skip, then let cooking catch up.
        self._kitchen.skipped += max(0.0, float(seconds))
        self._kitchen.update(force=True)


class RoboCasaObserver(Observer):
    def __init__(self, kitchen: RoboCasaKitchen):
        self._kitchen = kitchen

    def observe(self) -> Observation:
        return Observation(text=self._kitchen.describe())


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"
