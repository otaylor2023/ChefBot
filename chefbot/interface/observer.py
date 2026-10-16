"""Observer: what a step checker can look at (AGENT_SPEC §5.1).

Env-provided, like Detector. Its output goes only to the StateChecker and the
run log, never into the agent's LLM context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    text: str | None = None  # relations, one per line ("egg: inside pan")
    images: list[np.ndarray] | None = None  # RGB, for a VLM judge later
    extra: dict[str, Any] = field(default_factory=dict)  # proprioception, force, ...


class Observer(ABC):
    @abstractmethod
    def observe(self) -> Observation: ...
