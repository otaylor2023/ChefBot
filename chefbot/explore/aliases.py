"""Near-synonyms in the object word list, as one name: a static table (explore/config.yaml, aliases),
from what the words mean, not from any scene. The detector reports a group's name; the scorer and
the accuracy report compare names by group, so "cup" and "mug" are the same thing to them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out = {}
    for name, members in (cfg.get("aliases") or {}).items():
        for m in [name, *members]:
            out[m] = name
    return out


def canonical(label: str) -> str:
    """The group name of a label (a label in no group is its own)."""
    label = label.replace("_", " ")
    return _table().get(label, label)
