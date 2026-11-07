"""Agent-layer configuration: an agent's agent.yaml (chefbot/agents/<name>/), its `settings:` and `loop:`.
With no path, the default agent's (chefbot/agents/kitchen_planner)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "agents" / "kitchen_planner" / "agent.yaml"


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
        raw = dict(yaml.safe_load(Path(path or DEFAULT_PATH).read_text()).get("settings") or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in agent.yaml settings: {sorted(unknown)}")
        raw["default_vocab"] = tuple(raw.get("default_vocab", ()))
        return cls(**raw)


@dataclass(frozen=True)
class LoopConfig:
    """The agent loop's settings: `loop:` in agent.yaml (AGENT_SPEC §10)."""

    model: str = "anthropic/claude-sonnet-5.5"
    judge_model: str = "anthropic/claude-haiku-5.5"
    provider: dict | None = None  # OpenRouter provider routing, e.g. {"order": ["anthropic"], "allow_fallbacks": false}
    temperature: float | None = None  # None: the provider's default
    judge_temperature: float = 0.0
    time_mode: str = "paused"  # "paused" | "realtime"
    window_n: int = 8
    max_reprompts: int = 2
    stale_after_calls: int = 5
    stale_move_m: float = 0.05
    stale_turn_deg: float = 5.0
    clear_move_m: float = 0.5
    clear_turn_deg: float = 30.0
    empty_burner_s: float = 30.0
    max_turns: int = 150
    max_llm_cost_usd: float = 5.0
    max_sim_time_s: float = 1800.0
    max_wall_time_s: float = 3600.0
    max_consecutive_errors: int = 5
    log_full_requests: bool = True
    tools_exclude: tuple = ()  # tools never offered (on top of what the robot cannot do)
    tools_only: tuple = ()  # if set, only these tools (and the loop's plan / time tools if listed)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "LoopConfig":
        raw = dict(yaml.safe_load(Path(path or DEFAULT_PATH).read_text()).get("loop") or {})
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in agent.yaml loop: {sorted(unknown)}")
        for key in ("tools_exclude", "tools_only"):
            if key in raw:
                raw[key] = tuple(raw[key] or ())
        if raw.get("time_mode", "paused") not in ("paused", "realtime"):
            raise ValueError("loop.time_mode must be paused or realtime")
        return cls(**raw)
