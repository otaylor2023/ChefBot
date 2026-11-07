"""Symbolic dry run of a Recipe: applies each step's effects (from verbs.yaml) to a predicted world state and
reports what can't be true. Not physics, not food safety: `cooked` is a rule effect, not doneness.

run(recipe) for a dry run; start(recipe) + apply(step) + finish() to go one step at a time. For execution
monitoring: apply a step, observe(...) what the robot saw, diff() predicted against observed.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from .schema import Recipe, Step, Input

EPS = 1e-9


@dataclass
class Violation:
    code: str
    step: str | None
    message: str


@dataclass
class Fact:
    properties: dict[str, Any] = field(default_factory=dict)
    available: float = 1.0                                   # share of the item not yet consumed
    composition: dict[str, float] = field(default_factory=dict)   # base ingredient -> amount inside
    quantity_known: bool = True                              # False once a vague amount touched it
    location: str | None = None
    cooked_components: set[str] = field(default_factory=set)
    contents: list[str] = field(default_factory=list)        # containers only


@dataclass
class WorldState:
    items: dict[str, Fact] = field(default_factory=dict)
    containers: dict[str, Fact] = field(default_factory=dict)


@dataclass
class EngineResult:
    predicted: WorldState
    observed: WorldState
    violations: list[Violation]      # the extraction (or recipe) can't be executed as written
    unresolved: list[Violation]      # facts the source leaves open; reported, never fatal
    executed: list[str]
    blocked: list[str]               # skipped because an upstream step failed
    trace: list[dict[str, Any]]
    warnings: list[Violation]        # omission signals: unused ingredients/outputs

    @property
    def ok(self) -> bool:
        return not self.violations


def _unresolved(v: Any) -> bool:
    return isinstance(v, dict) and "unresolved" in v


def _state(inp: Input) -> dict[str, Any]:
    return {r.key: (r.value if r.unresolved is None else {"unresolved": r.unresolved}) for r in inp.required_state}


def _fraction(inp: Input) -> float | None:
    """A usable fraction, or None when the amount is vague/missing/out of range (then it's unresolved)."""
    f = inp.fraction
    return f if f is not None and 0 < f <= 1 else None


class StateEngine:
    def __init__(self, verbs_path: str | Path | None = None):
        table = yaml.safe_load(Path(verbs_path or Path(__file__).with_name("verbs.yaml")).read_text())
        self.rules: dict[str, dict] = table["verbs"]
        self.heat_properties: list[str] = table.get("heat_properties", ["temp_c"])
        self.state_keys: set[str] = {k for r in self.rules.values() for k in r.get("sets", {})}
        self.predicted, self.observed = WorldState(), WorldState()

    # ---------- monitoring ----------
    def observe(self, *, items: dict[str, Fact] | None = None, containers: dict[str, Fact] | None = None):
        """Record what was actually observed. Never overwrites the prediction."""
        self.observed.items.update(deepcopy(items or {}))
        self.observed.containers.update(deepcopy(containers or {}))

    def diff(self) -> list[dict[str, Any]]:
        """Observed vs predicted, only for what was observed (unobserved = unknown, not a mismatch)."""
        out = []
        for kind in ("items", "containers"):
            predicted = getattr(self.predicted, kind)
            for ref, got in getattr(self.observed, kind).items():
                want = predicted.get(ref)
                if want is None:
                    out.append({"ref": ref, "field": "exists", "predicted": False, "observed": True})
                    continue
                out += [{"ref": ref, "field": k, "predicted": want.properties[k], "observed": v}
                        for k, v in got.properties.items() if k in want.properties and want.properties[k] != v]
        return out

    # ---------- dry run ----------
    def run(self, recipe: Recipe, *, selected_optional: set[str] | None = None) -> EngineResult:
        for step in self.start(recipe, selected_optional=selected_optional):
            self.apply(step)
        return self.finish()

    def start(self, recipe: Recipe, *, selected_optional: set[str] | None = None) -> list[Step]:
        """Reset state; return the steps that will run (default choice branches, selected optional steps)."""
        self.recipe = recipe
        self.ingredients = {i.id: i for i in recipe.ingredients}
        self.predicted = WorldState({i: Fact(composition={i: 1.0}) for i in self.ingredients},
                                    {c.id: Fact() for c in recipe.containers})
        self.observed = WorldState()
        self.violations: list[Violation] = []
        self.unresolved: list[Violation] = []
        self.executed: list[str] = []
        self.blocked: list[str] = []
        self.trace: list[dict[str, Any]] = []
        self._used: set[str] = set()       # consumed by any attempted step (incl. failed/blocked)
        self._poisoned: set[str] = set()   # outputs that never materialised because their step failed

        chosen: set[str] = set()
        for ch in recipe.choices:
            if ch.default is None:
                self.unresolved.append(Violation("unresolved_choice", None, f"{ch.id}: no branch selected"))
            elif not 0 <= ch.default < len(ch.alternatives):
                self.violations.append(Violation("invalid_choice", None, f"{ch.id}: default out of range"))
            else:
                chosen |= set(ch.alternatives[ch.default])
        in_choice = {sid for ch in recipe.choices for alt in ch.alternatives for sid in alt}
        selected_optional = selected_optional or set()
        return [s for s in recipe.steps
                if not (s.id in in_choice and s.id not in chosen) and not (s.optional and s.id not in selected_optional)]

    def apply(self, step: Step) -> list[Violation]:
        """Apply one step to the predicted state; returns the violations it raised."""
        items, containers = self.predicted.items, self.predicted.containers
        before = len(self.violations)

        def err(code, msg):
            self.violations.append(Violation(code, step.id, msg))

        def note(code, msg):
            self.unresolved.append(Violation(code, step.id, msg))

        self._used |= {i.ref for i in step.inputs if i.mode == "consume"}
        if any(i.ref in self._poisoned for i in step.inputs):   # don't pile errors onto an upstream failure
            self.blocked.append(step.id)
            self.trace.append({"step": step.id, "verb": step.verb, "status": "blocked"})
            if step.output:
                self._poisoned.add(step.output)
            return []

        if step.verb == "other":        # outside the verb table: keep the graph, change nothing, report it
            rule = {}
            note("unmodeled_verb", step.verb_raw or "other")
        else:
            rule = self.rules.get(step.verb)
            if rule is None:
                err("unknown_verb", step.verb)
                return self._fail(step, before)

        # ---- preconditions ----
        food = [i.ref for i in step.inputs if i.mode == "consume"]
        if rule.get("max_inputs") is not None and len(food) > rule["max_inputs"]:
            err("verb_arity", f"{step.verb} takes at most {rule['max_inputs']} food input(s), got {food}; "
                              "combining several is 'mix'")
        if step.container and step.container not in containers:
            err("unknown_container", step.container)
        if "dest_container" in rule.get("needs", []) and not step.container:
            err("missing_container", "destination container required")
        if "heat_source" in rule.get("needs", []) and not any(
                i.mode == "require" and i.ref in containers
                and any(containers[i.ref].properties.get(k) is not None for k in self.heat_properties)
                for i in step.inputs):
            err("missing_heat_source", f"requires a container with one of {self.heat_properties} set")
        for inp in step.inputs:
            target = items.get(inp.ref) or containers.get(inp.ref)
            if target is None:
                err("unknown_input", inp.ref)
                continue
            for k, want in _state(inp).items():
                if k not in self.state_keys:
                    err("unknown_state_key", f"{inp.ref}.{k}: no verb ever sets {k!r}; settable keys are "
                                             f"{sorted(self.state_keys)} (heating is checked automatically)")
                elif _unresolved(want):
                    note("unresolved_field", f"{inp.ref}.{k}: {want['unresolved']}")
                elif target.properties.get(k) != want:
                    err("state_precondition", f"{inp.ref}.{k} requires {want}, got {target.properties.get(k)}")
            if inp.mode == "consume":
                f = _fraction(inp)
                if inp.ref in containers:
                    err("cannot_consume_container", inp.ref)
                elif f is None:
                    note("unresolved_fraction", f"{inp.ref}: unknown allocation")
                elif f > target.available + EPS:
                    msg = f"{inp.ref}: {f} requested, {target.available:.3g} left"
                    err("overconsumption", msg) if target.quantity_known else note("unverifiable_quantity", msg)
        if step.until and step.until.kind == "unresolved":
            note("unresolved_until", step.until.raw or "unknown completion condition")
        if len(self.violations) != before:
            return self._fail(step, before)

        # ---- effects ----
        changes = {}
        for key, value in rule.get("sets", {}).items():
            if isinstance(value, str) and value.startswith("$param."):
                name = value.removeprefix("$param.")
                value = getattr(step.params, name, None)
                if value is None:   # never invent a value the source didn't give
                    value = {"unresolved": step.params.unresolved or f"{name} not stated"}
                    note("unresolved_field", f"{name}: {value['unresolved']}")
            changes[key] = value

        if rule.get("target") == "equipment":
            equipment = [i.ref for i in step.inputs if i.mode == "require" and i.ref in containers]
            if not equipment:
                err("missing_equipment", f"{step.verb} requires an equipment input")
                return self._fail(step, before)
            for ref in equipment:
                for key in rule.get("unsets", []):
                    containers[ref].properties.pop(key, None)
                containers[ref].properties.update(changes)
            self.executed.append(step.id)
            self.trace.append({"step": step.id, "verb": step.verb, "status": "ok",
                               "equipment": {r: dict(containers[r].properties) for r in equipment}})
            return []

        material = [i for i in step.inputs if i.mode == "consume" and i.ref in items]
        composition: dict[str, float] = {}
        cooked: set[str] = set()
        quantity_known = True
        for inp in material:            # draw down the quantity ledger
            part = items[inp.ref]
            cooked |= part.cooked_components
            f = _fraction(inp)
            if f is None:               # amount unknown: keep lineage, take nothing, stop enforcing conservation
                part.quantity_known = quantity_known = False
                for base, amount in part.composition.items():
                    composition[base] = composition.get(base, 0) + amount
                continue
            quantity_known &= part.quantity_known
            f = min(f, part.available)
            ratio = f / part.available if part.available > EPS else 0
            for base, amount in part.composition.items():
                composition[base] = composition.get(base, 0) + amount * ratio
                part.composition[base] = amount * (1 - ratio)
            part.available -= f
            if part.available <= EPS:
                for c in containers.values():
                    if inp.ref in c.contents:
                        c.contents.remove(inp.ref)

        if not step.output:
            self.executed.append(step.id)
            self.trace.append({"step": step.id, "verb": step.verb, "status": "ok", "consumed": [i.ref for i in material]})
            return []
        if step.output in items or step.output in containers:
            err("duplicate_output", step.output)
            return self._fail(step, before)

        props = [items[i.ref].properties for i in material]   # output keeps what all its inputs agree on
        inherited = {k: v for k, v in props[0].items() if all(d.get(k) == v for d in props[1:])} if props else {}
        inherited.pop("components", None)
        inherited.update(changes)
        if composition:
            inherited["components"] = sorted(b for b, a in composition.items() if a > EPS or not quantity_known)
        if rule.get("cooks"):
            cooked |= set(composition)
        location = step.container or next((items[i.ref].location for i in material if items[i.ref].location), None)
        items[step.output] = Fact(properties=inherited, composition=composition, location=location,
                                  quantity_known=quantity_known, cooked_components=cooked)
        if location in containers:
            containers[location].contents.append(step.output)
        self.executed.append(step.id)
        self.trace.append({
            "step": step.id, "verb": step.verb, "status": "ok",
            "consumed": {i.ref: (_fraction(i) if _fraction(i) is not None else "unresolved") for i in material},
            "required": [i.ref for i in step.inputs if i.mode == "require"],
            "output": step.output, "location": location,
            "properties": {k: v for k, v in inherited.items() if k != "components"},
            "components": sorted(composition), "cooked_components": sorted(cooked),
            "until": step.until.model_dump(exclude_none=True) if step.until else None,
            "container_contents": {location: list(containers[location].contents)} if location in containers else {},
        })
        return []

    def _fail(self, step: Step, before: int) -> list[Violation]:
        if step.output:
            self._poisoned.add(step.output)
        self.trace.append({"step": step.id, "verb": step.verb, "status": "failed",
                           "violations": [v.code for v in self.violations[before:]]})
        return self.violations[before:]

    def finish(self) -> EngineResult:
        recipe, items = self.recipe, self.predicted.items
        finals = list(recipe.final_items)
        warnings: list[Violation] = []
        skipped = {s.output: s for s in recipe.steps if s.output and s.id not in self.executed
                   and s.id not in self.blocked and s.output not in self._poisoned}
        for fid in list(finals):   # "garnish (optional) -> dish": with the option off, check the main input
            producer = skipped.get(fid)
            if fid in items or producer is None or not producer.optional:
                continue
            if any(i.ref in self._poisoned for i in producer.inputs):
                finals.remove(fid)
                continue
            base = [i.ref for i in producer.inputs if i.mode == "consume" and i.ref in items]
            if base:
                finals = [f for f in finals if f != fid] + base
                warnings.append(Violation("final_from_optional_step", producer.id,
                                          f"{fid} comes from optional step {producer.id}; checked {base} instead"))
        for fid in finals:
            if fid in self._poisoned:
                continue
            final = items.get(fid)
            if final is None:
                self.violations.append(Violation("missing_final_item", None, fid))
                continue
            for base, amount in final.composition.items():
                if ((amount > EPS or not final.quantity_known) and base in self.ingredients
                        and self.ingredients[base].needs_cooking and base not in final.cooked_components):
                    self.violations.append(Violation("uncooked_component", None, f"{base} remains raw in {fid}"))
        for i in recipe.ingredients:
            if not i.optional and i.id not in self._used and i.id not in finals:
                warnings.append(Violation("unused_ingredient", None, i.id))
        for s in recipe.steps:
            if s.output and s.output in items and s.output not in self._used and s.output not in finals \
                    and s.id in self.executed:
                warnings.append(Violation("unused_output", s.id, s.output))
        return EngineResult(self.predicted, self.observed, self.violations, self.unresolved,
                            self.executed, self.blocked, self.trace, warnings)
