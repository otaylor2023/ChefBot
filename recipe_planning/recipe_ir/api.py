"""The one function ChefBot calls. Everything else (LLM calls, repair, reports, evals) stays behind it."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .config import DEFAULT_RECIPE_MODEL, DEFAULT_REASONING_EFFORT
from .adapters import load_source, from_text
from .engine import Violation, WorldState
from .llm_extractor import BaseExtractor, make_extractor
from .recipe_pipeline import run_pipeline
from .schema import Recipe


@dataclass
class Interpretation:
    recipe: Recipe | None             # main path as steps; options as choices / alternative rules
    status: str                       # "valid" | "recipe_findings" | "extraction_errors"
    extraction_errors: list[Violation]   # don't execute: the plan doesn't match the recipe
    recipe_findings: list[Violation]  # the recipe itself has a gap (e.g. an ingredient never cooked)
    unresolved: list[Violation]       # things the robot must decide or sense (heat level, "until golden")
    predicted: WorldState | None      # expected state after the last step; compare against observations
    trace: list[dict[str, Any]]       # expected state after each step
    edges: list[tuple[str, str]]      # step ordering constraints; unconnected steps may run in parallel


def interpret(source: str, *, model: str = DEFAULT_RECIPE_MODEL, reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
              extractor: BaseExtractor | None = None) -> Interpretation:
    """source: a file path, a URL, or the recipe text itself."""
    looks_like_text = "\n" in source or len(source) > 300
    doc = from_text(source) if looks_like_text else load_source(source)
    result = run_pipeline(doc, extractor or make_extractor(model, reasoning_effort=reasoning_effort))
    v = result.final.validation
    e = v.engine
    return Interpretation(v.recipe, v.status, v.extraction_errors, v.recipe_findings,
                          e.unresolved if e else [], e.predicted if e else None, e.trace if e else [], v.edges)
