"""Validated Recipe IR -> action-ready instructions, plus a separate robot-feasibility report.

instructions(validation)  robot-independent. For every step: what is acted on (and which base ingredients that
                          is), equipment and the state it must be in, settings, where the result goes, how
                          completion is recognised and what senses it, which steps must come first, the stage it
                          can run in, the tools it needs, the rules that apply, and the questions the recipe
                          leaves open. Recipe level: inventory, stages, open questions.
feasibility(instr, skills) what one skill profile (skills/*.yaml) can do with those instructions. Changing the
                          profile never changes the instructions.

Deterministic: no LLM. Everything is derived from the extraction, the state engine trace and the dependency edges.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import yaml
from .recipe_pipeline import Validation

ROOT = Path(__file__).resolve().parent
TOOLS = yaml.safe_load((ROOT / "tools.yaml").read_text())
SKILLS_DIR = ROOT.parent / "skills"

SENSE = {"time": "timer", "temp": "thermometer", "visual": "camera", "texture": "touch/probe",
         "unresolved": "ask a person or define a check"}


def _tool_patterns():
    pats = []
    for tool, words in TOOLS["lexicon"].items():
        for w in words:
            pats.append((len(w), tool, re.compile(r"\b" + re.escape(w).replace(r"\ ", r"\s+") + r"(e?s)?\b", re.I)))
    return sorted(pats, key=lambda p: -p[0])   # longest phrase wins ("wooden spoon" before "spoon")


_PATTERNS = _tool_patterns()


def mentioned_tools(text: str | None) -> list[str]:
    if not text:
        return []
    taken: list[tuple[int, int]] = []
    found: list[str] = []
    for _, tool, rx in _PATTERNS:
        for m in rx.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            taken.append(m.span())
            if tool not in found:
                found.append(tool)
    return found


def implied_tools(verb: str, style: str | None) -> list[str]:
    imp = TOOLS["implied"]
    if style and f"{verb}/{style.lower()}" in imp:
        return list(imp[f"{verb}/{style.lower()}"])
    return list(imp.get(verb, []))


def _stages(step_ids: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Longest-path depth in the dependency DAG: steps with the same stage have no ordering between them."""
    preds: dict[str, list[str]] = {s: [] for s in step_ids}
    for a, b in edges:
        if a in preds and b in preds:
            preds[b].append(a)
    depth: dict[str, int] = {}

    def d(s, seen=()):
        if s in depth:
            return depth[s]
        if s in seen:          # a cycle would be an extraction bug; don't recurse forever
            return 0
        depth[s] = 1 + max((d(p, seen + (s,)) for p in preds[s]), default=-1)
        return depth[s]

    for s in step_ids:
        d(s)
    return depth


def instructions(v: Validation) -> dict[str, Any]:
    r = v.recipe
    if r is None:
        return {"status": v.status, "error": v.schema_error, "steps": []}
    trace = {t["step"]: t for t in (v.engine.trace if v.engine else [])}
    unresolved = v.engine.unresolved if v.engine else []
    ingredients = {i.id: i for i in r.ingredients}
    containers = {c.id: c for c in r.containers}
    components: dict[str, list[str]] = {i: [i] for i in ingredients}
    for t in trace.values():
        if t.get("output"):
            components[t["output"]] = t.get("components", [])
    step_ids = [s.id for s in r.steps]
    stage = _stages(step_ids, v.edges)
    after: dict[str, list[str]] = {s: [] for s in step_ids}
    for a, b in v.edges:
        if b in after:
            after[b].append(a)
    rules_for: dict[str, list[dict]] = {}
    for rule in r.rules:
        for sid in rule.applies_to or []:
            rules_for.setdefault(sid, []).append({"kind": rule.kind, "text": rule.text})
    in_choice = {sid: ch.id for ch in r.choices for alt in ch.alternatives for sid in alt}

    steps, all_tools = [], {}
    eq_state: dict[str, dict] = {}          # equipment state as the steps run (after this step if it changes it)
    for s in r.steps:
        t = trace.get(s.id, {})
        for ref, props in (t.get("equipment") or {}).items():
            eq_state[ref] = props
        acts_on, equipment = [], []
        for inp in s.inputs:
            if inp.mode == "consume":
                ing = ingredients.get(inp.ref)
                if inp.fraction is None:
                    amount = inp.fraction_raw or "not stated"
                elif inp.fraction == 1:
                    amount = (ing.quantity_raw if ing and ing.quantity_raw else "all")
                else:
                    amount = f"{inp.fraction:g} of " + (ing.quantity_raw if ing and ing.quantity_raw else "it")
                acts_on.append({"ref": inp.ref, "kind": "ingredient" if ing else "intermediate",
                                "name": ing.name if ing else None, "made_of": components.get(inp.ref, []),
                                "amount": amount})
            else:
                c = containers.get(inp.ref)
                equipment.append({"ref": inp.ref, "type": c.type if c else None,
                                  "must_be": {q.key: (q.value if q.unresolved is None else f"? {q.unresolved}")
                                              for q in inp.required_state},
                                  "state": dict(eq_state.get(inp.ref, {}))})
        settings = {k: getattr(s.params, k) for k in ("temp_c", "heat", "duration_s") if getattr(s.params, k) is not None}
        if s.until:
            done = {"kind": s.until.kind, "target": s.until.target, "raw": s.until.raw, "sensed_by": SENSE[s.until.kind]}
        elif s.params.duration_s:
            done = {"kind": "time", "target": s.params.duration_s, "raw": None, "sensed_by": "timer"}
        else:
            done = {"kind": "action_complete", "target": None, "raw": None, "sensed_by": "the action's own result"}
        questions = [x.message for x in unresolved if x.step == s.id]
        if s.params.unresolved and not any(s.params.unresolved in q for q in questions):
            questions.append(s.params.unresolved)
        tools = [{"tool": x, "how": "mentioned"} for x in mentioned_tools(s.evidence.quote)]
        named = {x["tool"] for x in tools}
        if not named:   # an implied tool only when the text names none (a "fork" mention beats an implied spoon)
            tools += [{"tool": x, "how": "implied"} for x in implied_tools(s.verb, s.style)]
        for x in tools:
            entry = all_tools.setdefault(x["tool"], {"tool": x["tool"], "how": x["how"], "steps": [], "optional_only": True})
            entry["steps"].append(s.id)
            entry["optional_only"] &= bool(s.optional or s.id not in trace)
            if x["how"] == "mentioned":
                entry["how"] = "mentioned"
        dest = containers.get(s.container) if s.container else None
        steps.append({
            "id": s.id, "verb": s.verb, "style": s.style, "verb_raw": s.verb_raw,
            "instruction": s.evidence.quote or f"[inferred] {s.evidence.reason or s.evidence.implied_by}",
            "provenance": s.evidence.prov,
            "optional": s.optional, "choice": in_choice.get(s.id),
            "on_main_path": s.id in trace, "engine_status": t.get("status", "not run (optional or other branch)"),
            "acts_on": acts_on, "equipment": equipment, "settings": settings,
            "result": ({"id": s.output, "properties": t.get("properties", {}), "made_of": components.get(s.output, []),
                        "cooked": t.get("cooked_components", [])} if s.output else None),
            "goes_to": ({"ref": s.container, "type": dest.type if dest else None} if s.container else None),
            "done_when": done, "after": sorted(after[s.id]), "stage": stage.get(s.id, 0),
            "tools": tools, "rules": rules_for.get(s.id, []), "open_questions": questions,
        })

    n_stages = 1 + max(stage.values(), default=-1)
    return {
        "title": r.title, "status": v.status,
        "final_items": r.final_items,
        "inventory": {
            "ingredients": [{"id": i.id, "name": i.name, "quantity": i.quantity_raw, "optional": i.optional,
                             "needs_cooking": i.needs_cooking} for i in r.ingredients],
            "containers": [{"id": c.id, "type": c.type} for c in r.containers],
            "tools": list(all_tools.values()),
        },
        "stages": [[s for s in step_ids if stage.get(s) == k] for k in range(n_stages)],
        "steps": steps,
        "recipe_rules": [{"kind": x.kind, "text": x.text} for x in r.rules if not x.applies_to],
        "open_questions": ([x.message for x in unresolved if x.step is None] + list(r.ambiguities)
                           + [f"{x.code}: {x.message}" for x in v.recipe_findings]),
        "counts": {"steps": len(steps), "main_path": sum(st["on_main_path"] for st in steps), "stages": n_stages,
                   "open_questions": sum(len(st["open_questions"]) for st in steps),
                   "completion_needs_sensing": sum(st["done_when"]["kind"] in ("visual", "texture", "temp", "unresolved")
                                                   for st in steps if st["on_main_path"])},
    }


# ------------------------------------------------------------------ robot feasibility (swappable)
def load_skills(name_or_path: str = "chefbot_sim") -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = SKILLS_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(p.read_text())


def feasibility(instr: dict, skills: dict) -> dict[str, Any]:
    table = skills["verbs"]
    per_step, blocked_verbs = [], {}
    for st in instr["steps"]:
        if not st["on_main_path"]:
            continue
        key = f"{st['verb']}/{(st['style'] or '').lower()}"
        rule = table.get(key) or table.get(st["verb"]) or {"status": "unsupported", "note": "not in the profile"}
        per_step.append({"id": st["id"], "verb": st["verb"], "style": st["style"], "status": rule["status"],
                         "primitives": rule.get("primitives", []), "note": rule.get("note")})
        if rule["status"] != "supported":
            blocked_verbs.setdefault(st["verb"], []).append(st["id"])
    n = len(per_step)
    count = {k: sum(p["status"] == k for p in per_step) for k in ("supported", "partial", "unsupported")}
    first_gap = next((p["id"] for p in per_step if p["status"] != "supported"), None)
    return {"profile": skills.get("name"), "profile_version": skills.get("version"), **count, "steps": n,
            "supported_share": round(count["supported"] / n, 3) if n else 0.0,
            "fully_supported": n > 0 and count["supported"] == n,
            "first_gap": first_gap, "gaps_by_verb": blocked_verbs, "per_step": per_step}
