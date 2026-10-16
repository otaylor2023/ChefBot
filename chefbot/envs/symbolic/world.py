"""A symbolic kitchen: everything the RoboCasa env provides the agent, without physics.

Fixtures (counters, a stove with burners, a fridge, cabinets, appliances) stand
along the walls at fixed spots; objects are on or inside a fixture, a burner or
another object (an egg inside a pan on a burner), or held. Time only passes
through `advance()`: food heats, burners and appliances keep their state.

This is an env, like RoboCasa: the agent reaches it only through RobotAPI,
Detector, Clock and Observer (robot.py, perception.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

KITCHENS = Path(__file__).resolve().parent / "kitchens"

# What a place accepts, by kind. Fixtures and objects not listed accept anything.
COOKWARE = {"pan", "pot", "kettle"}
FOOD = {"egg", "cracked egg", "bread", "bacon", "pancake", "sausage", "tomato", "onion", "cheese", "butter", "milk",
        "water", "rice", "pasta"}  # fmt: skip
# Handheld tools: an egg cracker holds an egg and cracks it into what it is over when
# squeezed; an electric whisk runs while squeezed and beats what it is lowered into.
CRACKER, WHISK = "egg cracker", "whisk"
SHELL, TRASH = "eggshell", "trash can"  # cracking leaves the shell in the cracker; it goes to the trash (or scraps)
UTENSILS = {"spoon", "spatula"}  # stirred by hand (stir): mixes in a bowl, scrambles in a hot pan
LOWER_TARGETS = {"bowl", "pan", "pot", "mug", "plate"}  # containers a tool can be lowered into / over  # containers a tool can be lowered into / over
ACCEPTS = {
    "burner": COOKWARE,
    "toaster": {"bread"},
    "pot": FOOD,
    "plate": FOOD | {"fork", "knife", SHELL},
    "bowl": FOOD | {SHELL},
    "pan": FOOD | {SHELL},
    CRACKER: {"egg"},
    TRASH: None,  # anything
    "bowl": FOOD,
    "mug": {"water", "milk", "coffee"},
    "sink": None,  # anything
}
CONTAINERS = {"fridge", "cabinet", "microwave", "drawer", "toaster", "pan", "pot", "bowl", "mug", "kettle", "sink",
              CRACKER, TRASH}  # fmt: skip
OPENABLE = {"fridge", "cabinet", "microwave", "drawer", "oven"}
POWERED = {"toaster", "microwave", "kettle", "coffee_machine", "oven"}
UNPICKABLE = {"counter", "table", "stove", "burner", "fridge", "cabinet", "sink", "microwave", "toaster",
              "oven", "drawer", "coffee_machine", "shelf"}  # fmt: skip
SLOT = 0.25  # m between objects on a surface

TIMING_PATH = Path(__file__).resolve().parent / "timing.yaml"
NOISE_PATH = Path(__file__).resolve().parent / "noise.yaml"


@dataclass
class Timing:
    """How long things take (timing.yaml; a task can override parts of it)."""

    robot: dict
    cooking: dict  # food -> ((stage, seconds of heat), ...) in order
    beaten_after: float
    spoon_rate: float
    scrambled_after: float
    empty_burner_s: float
    toaster_cycle: float

    @classmethod
    def load(cls, overrides: dict | None = None) -> "Timing":
        raw = yaml.safe_load(TIMING_PATH.read_text())
        for section, values in (overrides or {}).items():
            if isinstance(values, dict) and isinstance(raw.get(section), dict):
                raw[section] = {**raw[section], **values}
            else:
                raw[section] = values
        cooking = {food: tuple(sorted(stages.items(), key=lambda kv: kv[1])) for food, stages in raw["cooking"].items()}
        return cls(dict(raw["robot"]), cooking, float(raw["whisk"]["beaten_after"]), float(raw["whisk"]["spoon_rate"]),
                   float(raw["whisk"]["scrambled_after"]), float(raw["safety"]["empty_burner_s"]),
                   float((raw.get("appliances") or {}).get("toaster_cycle", 150.0)))  # fmt: skip

    def stage(self, kind: str, heat: float) -> str:
        stage = "raw"
        for name, threshold in self.cooking.get(kind, self.cooking["default"]):
            if heat > threshold:
                stage = name
        return stage


SPOILED = {"overcooked", "burnt"}


@dataclass
class Fixture:
    name: str  # map name, e.g. "counter_1"; its label is the name minus "_n"
    kind: str
    location: np.ndarray  # (2,) middle of its top
    facing: np.ndarray  # (2,) unit vector into the room
    width: float = 0.6
    height: float = 0.92
    open: bool | None = None  # openable fixtures: True / False
    powered: bool | None = None  # appliances: on / off
    power_since: float = 0.0

    @property
    def label(self) -> str:
        return label_of(self.name)

    @property
    def across(self) -> np.ndarray:
        return np.array([-self.facing[1], self.facing[0]])


@dataclass
class Burner:
    name: str  # "burner_1"
    stove: str
    position: np.ndarray  # (3,)
    on: bool = False
    on_since: float = 0.0
    empty_since: float | None = None  # lit with nothing on it since


@dataclass
class Thing:
    """A loose object."""

    name: str  # unique, e.g. "egg" or "plate_2"
    kind: str  # its label: "egg", "plate"
    where: tuple[str, str] | None  # ("on" | "in", place name); None while held
    slot: float = 0.0  # offset along its fixture's width
    heat: float = 0.0  # seconds of heat so far (food)
    stage: str = "raw"
    whisked: float = 0.0  # seconds of whisking (cracked eggs)
    stirred_hot: float = 0.0  # seconds stirred while being heated (scrambling)
    splattered: bool = False  # a running whisk was lifted out of it


@dataclass
class Rates:
    """Injected failures: probabilities per attempt."""

    grasp: float = 0.0  # pick: the gripper closes on nothing
    place: float = 0.0  # place: the object ends up next to the target, not on it
    nav: float = 0.0  # go_to: blocked
    detect_miss: float = 0.0  # detect: a visible object is missed
    actuate: float = 0.0  # open / close: the gripper slips off the handle (retry)
    power: float = 0.0  # turn_on / turn_off: the knob or button does not take (retry)
    tool: float = 0.0  # lower_into, squeeze, stir, pour: the tool slips or jams (retry)
    crack_early: float = 0.0  # an egg cracks as it goes into the egg cracker, and is lost

    @classmethod
    def for_run(cls, task_failures: dict | None, noise: str = "task") -> "Rates":
        """`noise`: "off" (no random failures), "task" (the task file's), "on" (the
        default profile, noise.yaml, with the task file's on top)."""
        if noise == "off":
            return cls()
        rates = dict(yaml.safe_load(NOISE_PATH.read_text())) if noise == "on" else {}
        rates.update(task_failures or {})
        return cls(**rates)


def label_of(name: str) -> str:
    """As the agent's map reads names: egg_cracker_2 -> egg cracker."""
    head, sep, tail = name.rpartition("_")
    return (head if sep and tail.isdigit() else name).replace("_", " ")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


@dataclass
class World:
    fixtures: dict[str, Fixture]
    burners: dict[str, Burner]
    things: dict[str, Thing]
    t: float = 0.0
    held: str | None = None
    timing: Timing = field(default_factory=Timing.load)
    events: list[str] = field(default_factory=list)  # safety messages not yet collected
    _overcooked_reported: set = field(default_factory=set)
    _scheduled: list = field(default_factory=list)  # (time, callable): disturbances at a run time
    tool_into: str | None = None  # the held tool is lowered into / over this container
    squeezing: bool = False  # the held tool's trigger / handles are squeezed
    stirring: bool = False  # the robot is stirring with the held utensil (during a stir motion)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, kitchen: str | Path = "default", objects: list[dict] = (), timing: dict | None = None) -> "World":
        path = Path(kitchen) if Path(kitchen).suffix else KITCHENS / f"{kitchen}.yaml"
        raw = yaml.safe_load(path.read_text())
        fixtures, burners = {}, {}
        objects = [{("on" if k is True else k): v for k, v in o.items()} for o in objects]
        for name, spec in raw["fixtures"].items():
            kind = spec.get("kind", label_of(name))
            fixture = Fixture(
                name, kind, np.array(spec["location"], dtype=float), _unit(spec.get("facing", [0, 1])),
                float(spec.get("width", 0.6)), float(spec.get("height", 0.92)),
                open=False if kind in OPENABLE else None, powered=False if kind in POWERED else None,
            )  # fmt: skip
            fixtures[name] = fixture
            # Burners are numbered in the order listed (burner_1, burner_2, ...); each spot
            # (front_left: toward the room, on the left facing the stove) places it.
            for spot in spec.get("burners") or []:
                depth = -0.12 if spot.startswith("back") else 0.12
                side = 0.15 if spot.endswith("left") else -0.15 if spot.endswith("right") else 0.0
                xy = fixture.location + depth * fixture.facing + side * fixture.across
                burner = f"burner_{len(burners) + 1}"
                burners[burner] = Burner(burner, name, np.array([*xy, fixture.height + 0.01]))
        world = cls(fixtures, burners, {}, timing=Timing.load(timing))
        pending = list(objects)
        while pending:  # holders first: an egg in the egg cracker needs the cracker
            ready = [o for o in pending if world.is_place(o.get("in") or o.get("on") or "")]
            if not ready:
                world.add(pending[0])  # raises: unknown place
            for spec in ready:
                world.add(spec)
                pending.remove(spec)
        return world

    def add(self, spec: dict) -> Thing:
        """An object from a task file: {name, kind?, on: place | in: place}."""
        name = spec["name"]
        if name in self.things:
            raise ValueError(f"duplicate object {name}")
        kind = spec.get("kind", label_of(name))
        relation = "in" if "in" in spec else "on"
        place = spec.get(relation)
        if place is None or not self.is_place(place):
            raise ValueError(f"object {name}: unknown place {place!r}")
        thing = Thing(name, kind, (relation, place))
        thing.slot = self._free_slot(place, None)
        self.things[name] = thing
        if spec.get("heat"):
            thing.heat = float(spec["heat"])
            thing.stage = self.timing.stage(kind, thing.heat)
        return thing

    # -- places and positions -------------------------------------------------

    def is_place(self, name: str) -> bool:
        return name in self.fixtures or name in self.burners or name in self.things

    def kind_of(self, place: str) -> str:
        if place in self.fixtures:
            return self.fixtures[place].kind
        if place in self.burners:
            return "burner"
        return self.things[place].kind

    def fixture_of(self, place: str) -> Fixture | None:
        """The fixture a place (or an object) is ultimately on or in."""
        seen = set()
        while place not in self.fixtures:
            if place in seen:
                return None
            seen.add(place)
            if place in self.burners:
                return self.fixtures[self.burners[place].stove]
            thing = self.things.get(place)
            if thing is None or thing.where is None:
                return None
            place = thing.where[1]
        return self.fixtures[place]

    def position(self, name: str, robot_xy: np.ndarray | None = None) -> np.ndarray:
        """(3,) world position of a fixture, burner or object (held: in front of the robot)."""
        if name in self.fixtures:
            f = self.fixtures[name]
            return np.array([*f.location, f.height])
        if name in self.burners:
            return self.burners[name].position.copy()
        thing = self.things[name]
        if thing.where is None:
            xy = robot_xy if robot_xy is not None else np.zeros(2)
            return np.array([xy[0], xy[1], 1.0])
        relation, place = thing.where
        if place in self.fixtures:
            f = self.fixtures[place]
            xy = f.location + thing.slot * f.across
            z = f.height + (0.3 if f.kind in ("fridge", "cabinet") else 0.03)
            return np.array([*xy, z])
        base = self.position(place, robot_xy)
        return base + [0.0, 0.0, 0.03]

    def contents(self, place: str) -> list[Thing]:
        return [t for t in self.things.values() if t.where is not None and t.where[1] == place]

    def enclosed(self, name: str) -> bool:
        """Inside a closed fixture (fridge, cabinet, microwave), at any depth."""
        place = name
        while True:
            thing = self.things.get(place)
            if thing is None or thing.where is None:
                return False
            relation, place = thing.where
            fixture = self.fixtures.get(place)
            if fixture is not None:
                return relation == "in" and fixture.open is False

    def _free_slot(self, place: str, near: float | None) -> float:
        fixture = self.fixtures.get(place)
        if fixture is None or fixture.kind == TRASH:
            return 0.0  # objects, and the trash can: no spots to run out of
        used = [t.slot for t in self.contents(place)]
        half = max(0.0, fixture.width / 2 - SLOT / 2)
        candidates = np.arange(-half, half + 1e-9, SLOT) if half > 0 else np.array([0.0])
        free = [c for c in candidates if all(abs(c - u) > SLOT * 0.9 for u in used)]
        if not free:
            return math.nan
        target = 0.0 if near is None else near
        return float(min(free, key=lambda c: abs(c - target)))

    # -- changes (the robot's effects, and disturbances) -------------------------

    def can_place(self, obj: str, place: str) -> str | None:
        """Why `obj` cannot go on / in `place`, or None."""
        kind, place_kind = self.things[obj].kind, self.kind_of(place)
        if place == obj or self._is_inside(place, obj):
            return f"cannot put the {kind} on itself"
        accepts = ACCEPTS.get(place_kind, "any")
        if accepts is not None and accepts != "any" and kind not in accepts:
            return f"a {kind} does not go on the {place_kind}"
        if place in self.burners and self.contents(place):
            return f"{place} already has the {self.contents(place)[0].kind} on it"
        if place_kind == CRACKER and any(t.kind == SHELL for t in self.contents(place)):
            return "the egg cracker still has an eggshell in it: empty it first (pour it into the trash or a scraps bowl)"
        if place_kind == CRACKER and self.contents(place):
            return "the egg cracker already has an egg in it: crack that one first"
        fixture = self.fixtures.get(place)
        if fixture is not None and fixture.open is False:
            return f"the {fixture.kind} is closed"
        if fixture is not None and math.isnan(self._free_slot(place, None)):
            return f"there is no free spot on the {fixture.kind}"
        return None

    def move(self, obj: str, place: str | None, near: float | None = None) -> None:
        """Put `obj` on / in `place` (None: into the gripper). Contents come along."""
        thing = self.things[obj]
        if place is None:
            thing.where = None
            self.held = obj
            self.tool_into, self.squeezing = None, False
            return
        if self.held == obj:
            self.held = None
            self.tool_into, self.squeezing = None, False  # letting go of the tool lets go of its trigger
        relation = "in" if self.kind_of(place) in CONTAINERS else "on"
        thing.slot = self._free_slot(place, near) if place in self.fixtures else 0.0
        thing.where = (relation, place)

    def remove(self, obj: str) -> None:
        if self.held == obj:
            self.held = None
        for inner in self.contents(obj):
            self.remove(inner.name)
        del self.things[obj]

    def _is_inside(self, place: str, obj: str) -> bool:
        while place in self.things:
            thing = self.things[place]
            if thing.where is None:
                return False
            if thing.where[1] == obj:
                return True
            place = thing.where[1]
        return False

    # -- handheld tools ---------------------------------------------------------------

    def whisk_running(self) -> bool:
        return self.held is not None and self.things[self.held].kind == WHISK and self.squeezing

    def lower_into(self, place: str) -> str | None:
        """Lower the held tool into / over `place`. Returns why not, or None."""
        if self.held is None:
            return "the gripper is not holding anything"
        if self.kind_of(place) not in LOWER_TARGETS:
            return f"cannot lower anything into the {self.kind_of(place)}"
        if place == self.held:
            return "cannot lower it into itself"
        self.lift_out()
        self.tool_into = place
        return None

    def lift_out(self) -> str | None:
        """Lift the held tool out. A running whisk splatters what it was in."""
        if self.tool_into is None:
            return None
        message = None
        if self.whisk_running():
            target = self.things.get(self.tool_into)
            if target is not None:
                target.splattered = True
            message = f"the running whisk splattered the {self.kind_of(self.tool_into)} as it came out"
            self.events.append(message)
        self.tool_into = None
        return message

    def squeeze(self) -> str:
        """Squeeze the held tool. Returns what happened (one sentence)."""
        held = self.things[self.held]
        self.squeezing = True
        if held.kind == CRACKER:
            eggs = [t for t in self.contents(held.name) if t.kind == "egg"]
            if not eggs:
                return "the egg cracker is empty"
            for egg in eggs:  # the shell stays in the cracker either way
                self.remove(egg.name)
                self.add({"name": self._fresh("eggshell"), "kind": SHELL, "in": held.name})
            if self.tool_into is None:
                self.events.append("the egg spilled: the cracker was not over a bowl or pan")
                return "the egg cracked and spilled: the cracker was not over anything; the shell is left in the cracker"
            for _ in eggs:
                self.add({"name": self._fresh("cracked_egg"), "kind": "cracked egg", "in": self.tool_into})
            return (f"cracked {len(eggs)} egg{'s' if len(eggs) > 1 else ''} into the {self.kind_of(self.tool_into)};"
                    " the shell is left in the cracker")  # fmt: skip
        if held.kind == WHISK:
            return "the whisk is running" + ("" if self.tool_into else " in the air")
        return f"squeezing the {held.kind} does nothing"

    def release(self) -> None:
        self.squeezing = False

    def stir_report(self) -> str:
        """What stirring is doing right now (for the stir motion's result)."""
        place = self.tool_into
        food = [t for t in self.contents(place) if t.kind in FOOD] if place else []
        if not food:
            return "stirred, but there is nothing in it to stir"
        hot = any(self._heated(t) for t in food)
        return f"{'scrambled' if hot else 'mixed'} what is in the {self.kind_of(place)}"

    def pour(self, place: str) -> str | None:
        """Pour the held container's contents into `place`. Returns why not, or None."""
        contents = self.contents(self.held)
        if not contents:
            return f"the {self.things[self.held].kind} is empty"
        for food in contents:
            why = self.can_place(food.name, place)
            if why is not None:
                return why
        for food in contents:
            self.move(food.name, place)
        return None

    def _fresh(self, base: str) -> str:
        n = 1
        while f"{base}_{n}" in self.things:
            n += 1
        return f"{base}_{n}"

    def set_power(self, name: str, on: bool) -> None:
        if name in self.burners:
            burner = self.burners[name]
            if burner.on != on:
                burner.on, burner.on_since = on, self.t
                burner.empty_since = self.t if on and not self.contents(name) else None
            return
        fixture = self.fixtures[name]
        if fixture.powered != on:
            fixture.powered, fixture.power_since = on, self.t

    # -- time ---------------------------------------------------------------------

    def schedule(self, t: float, fn) -> None:
        """Run `fn(world)` once run time reaches `t`."""
        self._scheduled.append((float(t), fn))
        self._scheduled.sort(key=lambda item: item[0])

    def advance(self, dt: float) -> None:
        """Let `dt` seconds pass: food heats, safety rules and scheduled disturbances run."""
        end = self.t + dt
        while self._scheduled and self._scheduled[0][0] <= end:
            t, fn = self._scheduled.pop(0)
            self._advance(max(0.0, t - self.t))
            fn(self)
        self._advance(max(0.0, end - self.t))

    def _advance(self, dt: float) -> None:
        if dt <= 0:
            return
        if self.whisk_running() and self.tool_into is not None:
            for food in self.contents(self.tool_into):
                if food.kind == "cracked egg":
                    food.whisked += dt
        if self.stirring and self.tool_into is not None:
            for food in self.contents(self.tool_into):
                if food.kind == "cracked egg":
                    food.whisked += dt * self.timing.spoon_rate
                    if self._heated(food):
                        food.stirred_hot += dt
        for thing in self.things.values():
            if thing.kind in FOOD and self._heated(thing):
                thing.heat += dt
                stage = self.timing.stage(thing.kind, thing.heat)
                if stage != thing.stage:
                    thing.stage = stage
                    if stage in SPOILED and thing.name not in self._overcooked_reported:
                        self._overcooked_reported.add(thing.name)
                        self.events.append(f"the {thing.kind} is {stage}")
        self.t += dt
        for fixture in self.fixtures.values():
            if fixture.kind == "toaster" and fixture.powered and self.t - fixture.power_since >= self.timing.toaster_cycle:
                fixture.powered = False  # it pops up: a toaster turns itself off at the end of its cycle
                self.events.append(f"the {fixture.name} popped up (its {self.timing.toaster_cycle:.0f} s cycle ended)")
        for burner in self.burners.values():
            if not burner.on:
                continue
            if self.contents(burner.name):
                burner.empty_since = None
            elif burner.empty_since is None:
                burner.empty_since = self.t
            elif self.t - burner.empty_since >= self.timing.empty_burner_s:
                burner.on, burner.empty_since = False, None
                self.events.append(
                    f"{burner.name} was on with nothing on it for {self.timing.empty_burner_s:.0f} s; the robot turned it off"
                )

    def _heated(self, thing: Thing) -> bool:
        place = thing.where[1] if thing.where else None
        if place is None:
            return False
        if place in self.fixtures:
            f = self.fixtures[place]
            return f.kind in ("toaster", "microwave", "oven") and bool(f.powered) and f.open is not True
        holder = self.things.get(place)
        if holder is None or holder.kind not in COOKWARE or holder.where is None:
            return False
        under = holder.where[1]
        return under in self.burners and self.burners[under].on

    def take_events(self) -> list[str]:
        events, self.events = self.events, []
        return events

    # -- ground truth as text ---------------------------------------------------

    def describe(self) -> str:
        """Relations, one per line: what the StateChecker and the run log see."""
        lines = [f"held: {self.held or 'nothing'}"]
        for thing in sorted(self.things.values(), key=lambda t: t.name):
            if thing.where is None:
                state = [f"lowered into {self.tool_into}"] if self.tool_into else []
                if thing.kind == WHISK:
                    state.append("running" if self.squeezing else "off")
                elif self.squeezing:
                    state.append("squeezed")
                lines.append(f"{thing.name}: held" + (f", {', '.join(state)}" if state else ""))
            else:
                relation, place = thing.where
                lines.append(f"{thing.name}: {'inside' if relation == 'in' else 'on'} {place}")
        for burner in self.burners.values():
            state = f"on (for {_clock(self.t - burner.on_since)})" if burner.on else "off"
            lines.append(f"{burner.name}: {state}")
        for fixture in self.fixtures.values():
            if fixture.open is not None:
                lines.append(f"{fixture.name}: {'open' if fixture.open else 'closed'}")
            if fixture.powered is not None:
                state = f"on (for {_clock(self.t - fixture.power_since)})" if fixture.powered else "off"
                lines.append(f"{fixture.name}: {state}")
        for thing in sorted(self.things.values(), key=lambda t: t.name):
            if thing.kind in FOOD and (thing.heat > 0 or thing.kind in self.timing.cooking):
                lines.append(f"{thing.name}: cook state = {thing.stage} ({_clock(thing.heat)} of heat)")
            if thing.kind == "cracked egg":
                beaten = "beaten" if thing.whisked >= self.timing.beaten_after else "not beaten"
                lines.append(f"{thing.name}: {beaten} ({_clock(thing.whisked)} whisked)")
                if thing.heat > 0:
                    texture = "scrambled" if thing.stirred_hot >= self.timing.scrambled_after else "set in one piece, not scrambled"
                    lines.append(f"{thing.name}: {texture} ({_clock(thing.stirred_hot)} stirred while cooking)")
            if thing.splattered:
                lines.append(f"{thing.name}: splattered")
            if thing.kind in FOOD and thing.where is not None and thing.where[1] in self.things and any(
                t.kind == SHELL for t in self.contents(thing.where[1])
            ):
                lines.append(f"{thing.name}: has eggshell in it")
        return "\n".join(lines)

    def map_dict(self) -> dict:
        """The semantic map (agent's WorldMap format): fixtures and the objects as they start."""
        def spot(xy, facing) -> dict:
            return {"location": [round(float(v), 3) for v in xy], "facing": [round(float(v), 3) for v in facing]}

        fixtures = {f.name: spot(f.location, f.facing) for f in self.fixtures.values()}
        # Burners are fixed parts: on the map, so they keep their names (ids) between detects.
        for burner in self.burners.values():
            fixtures[burner.name] = spot(burner.position[:2], self.fixtures[burner.stove].facing)
        objects = {}
        for thing in self.things.values():
            holder = self.fixture_of(thing.name)
            if holder is None:
                continue
            xy = self.position(thing.name)[:2]
            objects[thing.name] = {"location": [round(float(v), 3) for v in xy], "in": holder.name}
        return {"frame": "world", "fixtures": fixtures, "objects": objects}


def _clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"
