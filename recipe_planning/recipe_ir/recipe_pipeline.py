"""extract -> validate -> (guarded repair) -> result.

Findings are split in two:
- EXTRACTION errors: the representation is broken (bad refs, unfindable quotes, missing heat step). Repairable.
- RECIPE findings: the representation is fine and the recipe itself has a gap (egg never cooked). Reported, never
  "repaired" -- otherwise the repair loop teaches the model to invent steps whenever the validator complains.
A repair is accepted only if it strictly reduces extraction errors, does not worsen evidence, and every step it
adds is stated in the source with a verifiable quote.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .schema import Recipe, SourceDocument, IdNote, normalize_ids
from .engine import StateEngine, EngineResult, Violation
from .checks import (check_evidence, check_faithfulness, check_params, check_rules, classify_uncited, instruction_coverage,
                     dependency_edges)
from .llm_extractor import BaseExtractor, LLMCall

RECIPE_CODES = {"uncooked_component"}
EVIDENCE_WARNINGS = {"quote_case_mismatch"}


@dataclass
class Validation:
    recipe: Recipe | None = None      # after id clean-up; evidence spans filled in
    engine: EngineResult | None = None
    evidence: list[Violation] = field(default_factory=list)
    schema_error: str | None = None
    coverage: float = 0.0
    gaps: list[tuple[int, int]] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    id_notes: list[IdNote] = field(default_factory=list)

    @property
    def extraction_errors(self) -> list[Violation]:
        notes = [Violation(n.code, n.step, n.message) for n in self.id_notes if n.error]
        if self.schema_error:
            return notes + [Violation("schema_error", None, self.schema_error[:1500])]
        eng = [v for v in self.engine.violations if v.code not in RECIPE_CODES] if self.engine else []
        return notes + eng + [v for v in self.evidence if v.code not in EVIDENCE_WARNINGS]

    @property
    def warnings(self) -> list[Violation]:
        """Everything worth a look that doesn't make the extraction wrong."""
        return ([Violation(n.code, n.step, n.message) for n in self.id_notes if not n.error]
                + [v for v in self.evidence if v.code in EVIDENCE_WARNINGS]
                + (self.engine.warnings if self.engine else []))

    @property
    def recipe_findings(self) -> list[Violation]:
        return [v for v in self.engine.violations if v.code in RECIPE_CODES] if self.engine else []

    @property
    def status(self) -> str:
        if self.extraction_errors:
            return "extraction_errors"
        return "recipe_findings" if self.recipe_findings else "valid"

    def uncited(self, source_text: str) -> dict[str, list[str]]:
        """Uncited instruction sentences that look like actions or rules (commentary is dropped)."""
        out: dict[str, list[str]] = {"action": [], "rule": []}
        for a, b in self.gaps:
            for kind, sent in classify_uncited(source_text[a:b]):
                if kind in out:
                    out[kind].append(sent)
        return out

    def grounding_rate(self) -> float:
        """Share of `stated` elements whose quote was found verbatim in the source."""
        if not self.recipe:
            return 0.0
        r = self.recipe
        stated = [o.evidence for coll in (r.ingredients, r.containers, r.steps) for o in coll
                  if o.evidence.prov == "stated"]
        return sum(1 for ev in stated if ev.span is not None) / len(stated) if stated else 0.0

    def summary(self) -> dict[str, Any]:
        e = self.engine
        return {
            "status": self.status,
            "extraction_errors": [vars(v) for v in self.extraction_errors],
            "recipe_findings": [vars(v) for v in self.recipe_findings],
            "unresolved": [vars(v) for v in e.unresolved] if e else [],
            "warnings": [vars(v) for v in self.warnings],
            "grounding_rate": round(self.grounding_rate(), 3),
            "instruction_coverage": round(self.coverage, 3),
            "n_steps": len(self.recipe.steps) if self.recipe else 0,
            "n_rules": len(self.recipe.rules) if self.recipe else 0,
            "n_edges": len(self.edges),
        }


def validate(x: Recipe | None, source: SourceDocument, error: str | None = None) -> Validation:
    if x is None:
        return Validation(schema_error=error or "no output")
    recipe, notes = normalize_ids(x)
    engine = StateEngine()
    evidence = check_evidence(recipe, source) + check_faithfulness(recipe, engine.rules) + check_rules(recipe)
    evidence += check_params(recipe, source)
    result = engine.run(recipe)
    coverage, gaps = instruction_coverage(recipe, source)
    return Validation(recipe, result, evidence, None, coverage, gaps, sorted(dependency_edges(recipe, engine.rules)), notes)


REPAIR_HINTS = {
    "missing_heat_source": "add a separate heat (stove) or preheat (oven) step BEFORE this step, whose only input is "
                           "the container as require; then give this step that container as a require input "
                           "(required_state []). See HEATING.",
    "missing_equipment": "an equipment step's only input is the container, mode require, required_state []; move any "
                         "food it consumes to the food step (e.g. melt) that follows it.",
    "heat_state_required": "delete this required_state entry: never require `heat`; being heated is checked "
                           "automatically from the earlier heat step.",
    "unstated_param": "set the value to null and describe the gap in params.unresolved; the heat step stays.",
    "verb_arity": "this verb acts on one food; use mix for combining two or more (whisk eggs with salt = mix/whisk).",
    "unknown_input": "every ref must be a declared ingredient, a declared container, or an earlier step's output id.",
}


def _hint(x) -> str:
    import os
    if os.environ.get("RECIPE_REPAIR_HINTS", "1") == "0":   # ablation: validator findings without fix instructions
        return ""
    code = x.code
    if code == "state_precondition" and ".heat requires" in x.message:
        code = "heat_state_required"
    if code == "state_precondition" and ".temp_c requires" in x.message:
        return "  -> the bake step's temp_c must equal the preheat step's params.temp_c (or drop this required_state)"
    return f"  -> {REPAIR_HINTS[code]}" if code in REPAIR_HINTS else ""


def findings_text(v: Validation) -> str:
    lines = [f"EXTRACTION {x.code} step={x.step}: {x.message}" + _hint(x) for x in v.extraction_errors]
    lines += [f"RECIPE {x.code}: {x.message}" for x in v.recipe_findings]
    lines += [f"WARNING {x.code} {x.step or ''}: {x.message}" for x in v.warnings]
    return "\n".join(lines) or "(none)"


def _step_sig(step) -> tuple:
    return (step.verb, step.evidence.quote)


def judge_repair(before: Validation, after: Validation, prev: Recipe | None,
                 new: Recipe | None) -> tuple[bool, str]:
    if after.schema_error:
        return False, "rejected: repair output failed schema/compile"
    if len(after.extraction_errors) >= len(before.extraction_errors):
        return False, f"rejected: extraction errors {len(before.extraction_errors)} -> {len(after.extraction_errors)} (no improvement)"
    if len(after.evidence) > len(before.evidence):
        return False, f"rejected: evidence problems increased {len(before.evidence)} -> {len(after.evidence)}"
    if prev is not None and new is not None and after.recipe is not None:
        old = {_step_sig(s) for s in prev.steps}
        spans = {s.id: s.evidence.span for s in after.recipe.steps}
        for s in new.steps:
            if _step_sig(s) in old:
                continue
            if s.evidence.prov != "stated" or spans.get(s.id) is None:
                return False, f"rejected: added step {s.id} ({s.verb}) is not stated in the source"
    return True, f"accepted: extraction errors {len(before.extraction_errors)} -> {len(after.extraction_errors)}"


@dataclass
class Attempt:
    kind: str
    call: LLMCall
    validation: Validation


@dataclass
class PipelineResult:
    source: SourceDocument
    attempts: list[Attempt]
    final_index: int
    repair_decision: str

    @property
    def final(self) -> Attempt:
        return self.attempts[self.final_index]

    def usage_totals(self) -> dict[str, float]:
        return {"input_tokens": sum(a.call.usage.input_tokens for a in self.attempts),
                "output_tokens": sum(a.call.usage.output_tokens for a in self.attempts),
                "latency_s": round(sum(a.call.usage.latency_s for a in self.attempts), 2),
                "calls": len(self.attempts)}


def run_pipeline(source: SourceDocument, extractor: BaseExtractor, *, repair: bool = True) -> PipelineResult:
    """extract -> (one retry if the output was unusable) -> (one guarded repair if extraction errors remain).
    A retried extraction goes through the same repair step and the same acceptance rule as a first one."""
    call = extractor.extract(source)
    attempts = [Attempt("extract", call, validate(call.recipe, source, call.error))]
    if attempts[0].validation.schema_error and repair:
        retry = extractor.extract(source, retry=True, prev_error=call.error)
        attempts.append(Attempt("retry", retry, validate(retry.recipe, source, retry.error)))
        if attempts[-1].validation.schema_error:
            return PipelineResult(source, attempts, 0, "retry failed")
    base = len(attempts) - 1
    current = attempts[base]
    if not current.validation.extraction_errors:
        return PipelineResult(source, attempts, base, "retry succeeded" if base else "not needed")
    if not repair:
        return PipelineResult(source, attempts, base, "disabled")
    rcall = extractor.repair(source, current.call.recipe, findings_text(current.validation))
    attempts.append(Attempt("repair", rcall, validate(rcall.recipe, source, rcall.error)))
    ok, decision = judge_repair(current.validation, attempts[-1].validation, current.call.recipe, rcall.recipe)
    prefix = "retry, then repair " if base else ""
    return PipelineResult(source, attempts, len(attempts) - 1 if ok else base, prefix + decision)
