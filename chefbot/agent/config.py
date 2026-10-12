"""Agent-layer configuration (agent.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "agent.yaml"


@dataclass(frozen=True)
class AgentConfig:
    approach_distance: float = 1.0
    nearby_radius: float = 2.5
    max_align_distance: float = 2.0
    map_match_radius: float = 0.75
    refine_radius: float = 0.15
    default_vocab: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AgentConfig":
        raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in agent config: {sorted(unknown)}")
        raw["default_vocab"] = tuple(raw.get("default_vocab", ()))
        return cls(**raw)
