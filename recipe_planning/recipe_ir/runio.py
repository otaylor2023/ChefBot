"""Shared helpers for the CLI and eval: cost lookup and saving a run folder."""
from __future__ import annotations
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
import yaml
from .recipe_pipeline import PipelineResult
from .report import render

ROOT = Path(__file__).resolve().parents[1]


def cost_usd(model: str, input_tokens: int, output_tokens: int, pricing_path: Path = ROOT / "pricing.yaml") -> float | None:
    try:
        table = yaml.safe_load(pricing_path.read_text()) or {}
    except FileNotFoundError:
        return None
    p = table.get(model)
    if not p or p.get("input") is None or p.get("output") is None:
        return None
    return round(input_tokens / 1e6 * p["input"] + output_tokens / 1e6 * p["output"], 5)


def result_json(result: PipelineResult) -> dict:
    v = result.final.validation
    u = result.usage_totals()
    model = result.attempts[0].call.usage.model
    return {
        "summary": v.summary(),
        "repair_decision": result.repair_decision,
        "usage": {**u, "model": model, "cost_usd": cost_usd(model, u["input_tokens"], u["output_tokens"])},
        "attempts": [{"kind": a.kind, **a.call.to_json(), "summary": a.validation.summary()} for a in result.attempts],
        "recipe": v.recipe.model_dump(mode="json") if v.recipe else None,
        "edges": v.edges,
        "trace": v.engine.trace if v.engine else [],
        "coverage_gaps": [result.source.text[a:b] for a, b in v.gaps],
        "uncited": v.uncited(result.source.text),
            }


def save_run(result: PipelineResult, name: str, out_dir: Path) -> Path:
    model = re.sub(r"[^A-Za-z0-9._-]", "_", result.attempts[0].call.usage.model)
    run = out_dir / f"{name}__{model}__{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "source.txt").write_text(result.source.text, encoding="utf-8")
    (run / "result.json").write_text(json.dumps(result_json(result), indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    (run / "report.html").write_text(render(result, name), encoding="utf-8")
    return run
