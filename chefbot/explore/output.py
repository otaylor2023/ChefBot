"""The explored map: what the agent found, in the semantic map's own format.

Fixtures come from the map the agent started with (as given, minus the
footprints); objects are the ones the agent's map holds at the end, named the
way the map builder names them: by label, a lone label keeps its name and
repeats become label_1, label_2, ... in order of position.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from chefbot.agent.world.map import WorldMap, label_of


def numbered(labels: list[str]) -> list[str]:
    totals = {label: labels.count(label) for label in labels}
    seen: dict[str, int] = {}
    names = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        names.append(label if totals[label] == 1 else f"{label}_{seen[label]}")
    return names


def explored_map(start: dict, world_map: WorldMap) -> dict:
    fixtures = copy.deepcopy(start.get("fixtures") or {})
    for spec in fixtures.values():
        spec.pop("size", None)
    found = [
        (label_of(e.name), e.location, e.inside)
        for e in world_map.list()
        if e.kind == "object" and e.inside is not None
    ]
    found.sort(key=lambda o: (o[0], round(o[1][0], 2), round(o[1][1], 2)))
    objects = {}
    for name, (_, location, inside) in zip(numbered([o[0] for o in found]), found):
        while name in fixtures or name in objects:
            name += "_object"
        objects[name] = {"location": [round(float(location[0]), 3), round(float(location[1]), 3)], "in": inside}
    return {"frame": start.get("frame", "world"), "fixtures": fixtures, "objects": objects}


def save(semantic: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(semantic, sort_keys=False, default_flow_style=None))
