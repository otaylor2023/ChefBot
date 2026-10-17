"""The one recipe schema. The LLM writes it (OpenAI strict structured output), the validator reads it,
and it is what every saved run, gold file and fixture contains.

Strict-mode rules shape it: every field is required (nullable instead of optional), no free-form dicts,
extra="forbid". The verb enum comes from verbs.yaml. Quote positions (spans) are computed by code and kept in a
private attribute, so they never appear in the schema the model sees.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

VERB_TABLE = yaml.safe_load(Path(__file__).with_name("verbs.yaml").read_text())
VERB_NAMES = tuple(VERB_TABLE["verbs"]) + ("other",)
VerbName = Literal[VERB_NAMES]  # type: ignore[valid-type]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(_Strict):
    quote: str | None                 # exact substring of the source; code finds its position
    prov: Literal["stated", "inferred"]
    reason: str | None                # required when inferred
    implied_by: Literal["ingredient_list"] | None
    _span: tuple[int, int] | None = PrivateAttr(default=None)

    @property
    def span(self) -> tuple[int, int] | None:
        return self._span


class Ingredient(_Strict):
    id: str
    name: str
    quantity_raw: str | None
    needs_cooking: bool               # unsafe raw: egg, meat, poultry, fish
    optional: bool
    evidence: Evidence


class Container(_Strict):
    id: str
    type: str
    evidence: Evidence


class StateReq(_Strict):
    key: str                          # a key some verb sets (temp_c, prepared, form, ...)
    value: str | float | bool | None
    unresolved: str | None


class Input(_Strict):
    ref: str                          # ingredient, container or an earlier step's output
    mode: Literal["consume", "require"]   # consume food / require equipment state
    fraction: float | None            # of the item's original amount; null = not stated
    fraction_raw: str | None          # the source's words when the amount is vague
    required_state: list[StateReq]


class Params(_Strict):
    temp_c: float | None
    heat: str | None
    duration_s: float | None
    unresolved: str | None            # what the source left open ("heat level not stated")


class Until(_Strict):
    kind: Literal["time", "visual", "temp", "texture", "unresolved"]
    target: str | float | None
    raw: str | None


class Step(_Strict):
    id: str
    verb: VerbName
    verb_raw: str | None              # source wording when verb == "other"
    style: str | None                 # nuance below the verb: dice, fold, whisk
    inputs: list[Input]
    output: str | None                # new item id; None for equipment steps
    container: str | None             # where the output ends up
    params: Params
    until: Until | None
    optional: bool
    evidence: Evidence


class Choice(_Strict):
    id: str
    alternatives: list[list[str]]     # step ids each option needs
    default: int | None
    evidence: Evidence


class Rule(_Strict):
    """What the recipe says that isn't a step. Kept for the planner/monitor; the engine doesn't enforce it yet."""
    id: str
    kind: Literal["avoid", "contingency", "alternative", "repeat"]
    applies_to: list[str]             # step ids; empty = whole recipe
    text: str
    evidence: Evidence


class Recipe(_Strict):
    title: str
    ingredients: list[Ingredient]
    containers: list[Container]
    steps: list[Step]
    choices: list[Choice]
    rules: list[Rule]
    final_items: list[str]
    ambiguities: list[str]

    @model_validator(mode="before")
    @classmethod
    def _accept_pre_rules_outputs(cls, data):
        # Outputs saved before rules existed (v0.4 and earlier) have no `rules`. A before-validator doesn't
        # change the JSON schema, so the model is still required to write the field.
        if isinstance(data, dict) and "rules" not in data:
            data = {**data, "rules": []}
        return data

    def evidenced(self):
        """(id, evidence) for every element that carries evidence, in reading order."""
        for coll in (self.ingredients, self.containers, self.steps, self.choices, self.rules):
            for obj in coll:
                yield obj.id, obj.evidence


# ------------------------------------------------------------------ source text
class Section(BaseModel):
    kind: Literal["ingredients", "instructions", "other"]
    start: int
    end: int


class SourceDocument(BaseModel):
    text: str
    sections: list[Section] = Field(default_factory=list)
    structured_hints: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ id clean-up
@dataclass
class IdNote:
    code: str
    step: str | None
    message: str
    error: bool   # True: ambiguous (extraction error). False: fixed here, shown as a warning.


def normalize_ids(recipe: Recipe) -> tuple[Recipe, list[IdNote]]:
    """Fix id collisions that have one sensible reading; name the ones that don't. Returns a copy."""
    r = recipe.model_copy(deep=True)
    notes: list[IdNote] = []
    seen: Counter[str] = Counter()
    for st in r.steps:                                    # duplicate step id: rename (harmless)
        seen[st.id] += 1
        if seen[st.id] > 1:
            new = f"{st.id}_{seen[st.id]}"
            notes.append(IdNote("renamed_step_id", new, f"step id {st.id!r} used twice; renamed to {new!r}", False))
            st.id = new
    ingredient_ids: set[str] = set()
    for i in r.ingredients:                               # ingredient declared twice: references are ambiguous
        if i.id in ingredient_ids:
            notes.append(IdNote("duplicate_ingredient_id", None,
                                f"ingredient id {i.id!r} declared twice ({i.name!r}); references are ambiguous", True))
            i.id = f"{i.id}_2"
        ingredient_ids.add(i.id)
    container_ids: set[str] = set()
    for c in r.containers:
        if c.id in container_ids:
            notes.append(IdNote("duplicate_container_id", None, f"container id {c.id!r} declared twice", True))
        elif c.id in ingredient_ids:                      # container named like an ingredient: mode tells them apart
            new = f"{c.id}_container"
            notes.append(IdNote("renamed_container_id", None,
                                f"container id {c.id!r} equals an ingredient id; renamed to {new!r}", False))
            for st in r.steps:
                if st.container == c.id:
                    st.container = new
                for inp in st.inputs:
                    if inp.ref == c.id and inp.mode == "require":
                        inp.ref = new
            c.id = new
        container_ids.add(c.id)
    return r, notes
