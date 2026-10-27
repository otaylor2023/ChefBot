"""Object memory: each loose object as one instance with its own identity, what has
been done with it, its states, and what it is set aside for.

Several like things (five forks) are five instances, fork_1 .. fork_5, named as the
map names them. An instance keeps its name as it moves: when you pick it and put it
down, and when it is seen again, matched by where it was last known and, when the
detector gives one, by how it looks (an appearance signature, and attributes such
as a colour). Built only from the agent's own records (its tool calls, detections,
its notes), never from ground truth.

Queries go by logical category (categories.yaml: cutlery, cookware, ...), label,
state, action and recipe: "which fork was used to stir what", "a clean spoon not
set aside for anything".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from chefbot.agent.world.map import label_of

_CATEGORIES = yaml.safe_load((Path(__file__).parent / "categories.yaml").read_text())
OTHER = "other"

# What a successful action does to the states of the object acted with (`held`) and the
# object acted on (`target`), when they have that state key.
EFFECTS: dict[str, dict[str, dict[str, str]]] = {
    "stirred": {"held": {"clean": "dirty"}},
    "lowered_into": {"held": {"clean": "dirty"}},
    "squeezed": {"held": {"clean": "dirty"}},
    "poured_into": {"held": {"fill": "empty", "clean": "dirty"}, "target": {"fill": "full", "clean": "dirty"}},
    "cracked": {"held": {"prep": "used up"}},
}


def category_of(label: str, categories: dict | None = None) -> str:
    for name, spec in (categories or _CATEGORIES).items():
        if label in spec["labels"]:
            return name
    return OTHER


@dataclass
class StateFact:
    value: str
    source: str  # "you" (your action), "seen", "note" (you said so), "default" (before anything was known)
    t: float


@dataclass
class Use:
    t: float
    action: str  # picked, placed, poured_into, poured_from, stirred, stirred_in, lowered_into, squeezed, ...
    other: str | None = None  # the object or place on the other end: what it was poured into, put on, ...
    step: str | None = None  # the plan step it was done for
    recipe: str | None = None  # the plan goal (the dish)
    note: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Plan:
    """Set aside for a later use: a recipe step, and the ingredient it will hold or work on."""

    recipe: str | None = None
    step: str | None = None
    ingredient: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Instance:
    id: str  # its name on the map: fork_3
    label: str
    category: str
    attributes: dict[str, str] = field(default_factory=dict)  # what tells it from like ones: {"color": "red"}
    signature: np.ndarray | None = None  # unit appearance vector (mean of the sightings), if the detector gives one
    sightings: int = 0
    last_xy: tuple[float, float] | None = None
    last_where: str | None = None  # what it was last known on or in, or "gripper"
    last_t: float | None = None
    states: dict[str, StateFact] = field(default_factory=dict)
    uses: list[Use] = field(default_factory=list)
    plans: list[Plan] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)  # ingredients put in it (eggs in the bowl)
    gone: bool = False  # used up (an egg cracked), or thrown away

    def state(self, key: str) -> str | None:
        fact = self.states.get(key)
        return fact.value if fact is not None else None

    def used_for(self, action: str | None = None) -> list[Use]:
        """Uses other than being carried around (picked, placed)."""
        return [u for u in self.uses if u.action not in ("picked", "placed", "seen_moved")
                and (action is None or u.action == action)]  # fmt: skip

    def tag(self) -> str:
        """Short state line for the KITCHEN block: "dirty, empty; for s5 (eggs)"."""
        parts = [f.value for k, f in self.states.items() if f.source != "default"]
        if self.contents:
            parts.append("holds " + ", ".join(self.contents))
        text = ", ".join(parts)
        if self.plans:
            p = self.plans[0]
            text += ("; " if text else "") + "for " + (p.step or p.recipe or "later") + (f" ({p.ingredient})" if p.ingredient else "")
        return text

    def to_dict(self) -> dict:
        out: dict = {"label": self.label, "category": self.category}
        if self.attributes:
            out["attributes"] = dict(self.attributes)
        if self.signature is not None:
            out["signature"] = [round(float(v), 4) for v in self.signature]
        out["sightings"] = self.sightings
        if self.last_xy is not None:
            out["last_xy"] = [round(v, 3) for v in self.last_xy]
        for key in ("last_where", "last_t"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.states:
            out["states"] = {k: {"value": f.value, "source": f.source, "t": f.t} for k, f in self.states.items()}
        if self.uses:
            out["uses"] = [u.to_dict() for u in self.uses]
        if self.plans:
            out["plans"] = [p.to_dict() for p in self.plans]
        if self.contents:
            out["contents"] = list(self.contents)
        if self.gone:
            out["gone"] = True
        return out

    @classmethod
    def from_dict(cls, id: str, raw: dict) -> "Instance":
        sig = raw.get("signature")
        return cls(
            id=id, label=raw["label"], category=raw.get("category", category_of(raw["label"])),
            attributes=dict(raw.get("attributes") or {}),
            signature=np.asarray(sig, dtype=float) if sig is not None else None,
            sightings=int(raw.get("sightings", 0)),
            last_xy=tuple(raw["last_xy"]) if raw.get("last_xy") else None,
            last_where=raw.get("last_where"), last_t=raw.get("last_t"),
            states={k: StateFact(v["value"], v["source"], float(v["t"])) for k, v in (raw.get("states") or {}).items()},
            uses=[Use(**u) for u in raw.get("uses") or []],
            plans=[Plan(**p) for p in raw.get("plans") or []],
            contents=list(raw.get("contents") or []), gone=bool(raw.get("gone")),
        )  # fmt: skip


class ObjectMemory:
    def __init__(self, now: Callable[[], float] = lambda: 0.0, categories: dict | None = None):
        self.now = now
        self.categories = categories or _CATEGORIES
        self._items: dict[str, Instance] = {}
        # Set by the agent loop: {"step": active step id, "recipe": plan goal}, for the uses it records.
        self.context: Callable[[], dict] = lambda: {}

    # -- instances --------------------------------------------------------------

    def get(self, id: str) -> Instance | None:
        return self._items.get(id)

    def all(self) -> list[Instance]:
        return list(self._items.values())

    def ensure(self, id: str, label: str | None = None) -> Instance:
        """The instance named `id`, made (with its category's starting states) if new."""
        inst = self._items.get(id)
        if inst is None:
            label = label or label_of(id)
            category = category_of(label, self.categories)
            defaults = (self.categories.get(category) or {}).get("states") or {}
            inst = Instance(id, label, category, states={k: StateFact(v, "default", self.now()) for k, v in defaults.items()})
            self._items[id] = inst
        return inst

    def rename(self, old: str, new: str) -> None:
        """A closer look told it is another one: it keeps its record under the new name."""
        if old in self._items and new not in self._items:
            inst = self._items.pop(old)
            inst.id = new
            self._items[new] = inst

    # -- what happens to them ------------------------------------------------------

    def seen(self, id: str, label: str, xy, where: str | None, signature=None, attributes: dict | None = None,
             moved: bool = False) -> Instance:  # fmt: skip
        inst = self.ensure(id, label)
        inst.sightings += 1
        inst.last_xy = (float(xy[0]), float(xy[1]))
        inst.last_where, inst.last_t = where, self.now()
        if signature is not None:
            sig = np.asarray(signature, dtype=float)
            sig = sig / (np.linalg.norm(sig) or 1.0)
            if inst.signature is None:
                inst.signature = sig
            else:  # running mean of the sightings, kept unit length
                mean = inst.signature * (inst.sightings - 1) + sig
                inst.signature = mean / (np.linalg.norm(mean) or 1.0)
        if attributes:
            inst.attributes.update(attributes)
        if moved:  # not where you left it: someone else moved it
            self._log(inst, "seen_moved", where)
        return inst

    def did(self, action: str, id: str, other: str | None = None, note: str | None = None) -> None:
        """You did `action` with `id` (on `other`): logged on both, with the states it changes."""
        inst = self.ensure(id)
        self._log(inst, action, other, note)
        self._apply(inst, EFFECTS.get(action, {}).get("held", {}))
        if action == "placed":
            inst.last_where, inst.last_t = other, self.now()
        elif action == "picked":
            inst.last_where, inst.last_t = "gripper", self.now()
        if other is None or other not in self._items:
            return
        target = self._items[other]
        back = {"poured_into": "poured_from", "stirred": "stirred_in", "lowered_into": "had_lowered_in",
                "squeezed": "squeezed_into"}.get(action)  # fmt: skip
        if back:
            self._log(target, back, id, note)
        self._apply(target, EFFECTS.get(action, {}).get("target", {}))
        if action == "poured_into":
            for c in inst.contents:
                if c not in target.contents:
                    target.contents.append(c)
            inst.contents.clear()

    def put_in(self, ingredient: str, container: str) -> None:
        """An ingredient went into a container (an egg cracked into the bowl)."""
        inst = self.ensure(container)
        if ingredient not in inst.contents:
            inst.contents.append(label_of(ingredient) if ingredient in self._items else ingredient)
        self._apply(inst, {"fill": "some"} if inst.state("fill") == "empty" else {})
        if ingredient in self._items:
            ing = self._items[ingredient]
            self._log(ing, "added_to", container)

    def use_up(self, id: str) -> None:
        inst = self.ensure(id)
        inst.gone = True
        self._apply(inst, {"prep": "used up"})

    def set_state(self, id: str, key: str, value: str, source: str = "note") -> None:
        inst = self.ensure(id)
        inst.states[key] = StateFact(value, source, self.now())
        if key == "fill" and value == "empty":
            inst.contents.clear()

    def reserve(self, id: str, recipe: str | None = None, step: str | None = None,
                ingredient: str | None = None, note: str | None = None) -> None:  # fmt: skip
        inst = self.ensure(id)
        ctx = self.context()
        inst.plans.append(Plan(recipe or ctx.get("recipe"), step, ingredient, note))

    def release(self, id: str, step: str | None = None) -> None:
        """Not set aside any more (for `step`, or for anything)."""
        inst = self.ensure(id)
        inst.plans = [p for p in inst.plans if step is not None and p.step != step]

    def _log(self, inst: Instance, action: str, other: str | None, note: str | None = None) -> None:
        ctx = self.context()
        inst.uses.append(Use(self.now(), action, other, ctx.get("step"), ctx.get("recipe"), note))

    def _apply(self, inst: Instance, changes: dict[str, str]) -> None:
        for key, value in changes.items():
            if key in inst.states:
                inst.states[key] = StateFact(value, "you", self.now())

    # -- telling like ones apart ------------------------------------------------------

    def appearance_cost(self, id: str, signature=None, attributes: dict | None = None) -> float:
        """How unlike instance `id` a sighting looks: 0 the same, up to about 1 (0 when unknown)."""
        inst = self._items.get(id)
        if inst is None:
            return 0.0
        cost = 0.0
        if signature is not None and inst.signature is not None:
            sig = np.asarray(signature, dtype=float)
            cost += max(0.0, 1.0 - float(sig @ inst.signature) / (np.linalg.norm(sig) or 1.0))
        for k, v in (attributes or {}).items():
            if k in inst.attributes and inst.attributes[k] != v:
                cost += 0.5
        return cost

    def distinguishing(self, id: str) -> dict[str, str]:
        """The attributes of `id` that none of its like ones share (how to tell it apart)."""
        inst = self._items[id]
        others = [o for o in self._items.values() if o.label == inst.label and o.id != id]
        return {k: v for k, v in inst.attributes.items() if all(o.attributes.get(k) != v for o in others)}

    # -- queries ----------------------------------------------------------------------

    def find(self, category: str | None = None, label: str | None = None, state: dict[str, str] | None = None,
             action: str | None = None, recipe: str | None = None, step: str | None = None,
             free: bool | None = None, include_gone: bool = False) -> list[Instance]:  # fmt: skip
        """Instances matching every filter given. `action`: it was used that way (stirred, poured_into...).
        `recipe` / `step`: used or set aside for it. `free`: True only those set aside for nothing."""
        out = []
        for inst in self._items.values():
            if inst.gone and not include_gone:
                continue
            if category is not None and inst.category != category:
                continue
            if label is not None and inst.label != label.replace("_", " "):
                continue
            if state and any(inst.state(k) != v for k, v in state.items()):
                continue
            if action is not None and not inst.used_for(action):
                continue
            if recipe is not None and not any(recipe in (u.recipe or "") for u in inst.uses) \
                    and not any(recipe in (p.recipe or "") for p in inst.plans):  # fmt: skip
                continue
            if step is not None and not any(u.step == step for u in inst.uses) and not any(p.step == step for p in inst.plans):
                continue
            if free is not None and free == bool(inst.plans):
                continue
            out.append(inst)
        return sorted(out, key=lambda i: i.id)

    def usage(self, category: str | None = None, label: str | None = None) -> dict[str, list[str]]:
        """Which instance was used doing what: {"fork_2": ["stirred in bowl (s2)"], "fork_1": []}."""
        out = {}
        for inst in self.find(category=category, label=label, include_gone=True):
            out[inst.id] = [_use_text(u) for u in inst.used_for()]
        return out

    def describe(self, id: str) -> dict:
        inst = self._items[id]
        out = {"id": inst.id, "label": inst.label, "category": inst.category}
        if inst.attributes:
            out["attributes"] = dict(inst.attributes)
            if (mark := self.distinguishing(id)):
                out["tells_it_apart"] = mark
        if inst.last_where is not None:
            out["last_where"] = inst.last_where
        if inst.states:
            out["states"] = {k: f.value for k, f in inst.states.items()}
        if inst.contents:
            out["contents"] = list(inst.contents)
        out["uses"] = [_use_text(u) for u in inst.used_for()]
        if inst.plans:
            out["set_aside_for"] = [p.to_dict() for p in inst.plans]
        if inst.gone:
            out["gone"] = True
        return out

    def categories_of(self) -> dict[str, list[str]]:
        """Category -> the instances in it."""
        out: dict[str, list[str]] = {}
        for inst in sorted(self._items.values(), key=lambda i: i.id):
            if not inst.gone:
                out.setdefault(inst.category, []).append(inst.id)
        return out

    # -- saving -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"objects": {i.id: i.to_dict() for i in self._items.values()}}

    @classmethod
    def from_dict(cls, raw: dict, now: Callable[[], float] = lambda: 0.0) -> "ObjectMemory":
        mem = cls(now)
        for id, spec in (raw.get("objects") or {}).items():
            mem._items[id] = Instance.from_dict(id, copy.deepcopy(spec))
        return mem

    def merge(self, other: "ObjectMemory") -> None:
        """Take what `other` remembers (an earlier run) for each object; the rest stay as they are."""
        for id, inst in other._items.items():
            self._items[id] = copy.deepcopy(inst)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path, now: Callable[[], float] = lambda: 0.0) -> "ObjectMemory":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {}, now)


def _use_text(u: Use) -> str:
    text = u.action.replace("_", " ") + (f" {u.other}" if u.other else "")
    if u.step:
        text += f" ({u.step})"
    if u.note:
        text += f": {u.note}"
    return text
