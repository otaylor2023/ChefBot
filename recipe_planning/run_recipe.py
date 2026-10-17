#!/usr/bin/env python
"""Extract one recipe with an LLM, validate it, and print/save an inspectable report.

  python run_recipe.py --input recipes/scrambled_eggs.txt --model gpt-6.1-sol --reasoning-effort low --show-trace
  python run_recipe.py --input https://example.com/some-recipe --model claude-sonnet-5-5
  python run_recipe.py --input recipes/onion_egg.txt --replay recipes/gold/onion_egg.gold.json   # offline
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from recipe_ir.config import DEFAULT_RECIPE_MODEL, DEFAULT_REASONING_EFFORT
from recipe_ir.adapters import load_source
from recipe_ir.llm_extractor import make_extractor, ReplayExtractor
from recipe_ir.recipe_pipeline import run_pipeline
from recipe_ir.runio import save_run, result_json


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="recipe .txt/.html file or URL")
    ap.add_argument("--model", default=DEFAULT_RECIPE_MODEL, help="gpt-* / o* (OPENAI_API_KEY) or claude-* (ANTHROPIC_API_KEY)")
    ap.add_argument("--temperature", type=float, default=0.0, help="dropped automatically if the model rejects it")
    ap.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, help="OpenAI reasoning models only, e.g. low / medium / high")
    ap.add_argument("--max-output-tokens", type=int, default=8000, help="caps the cost of a runaway response")
    ap.add_argument("--replay", nargs="+", help="recorded responses instead of an API call: extraction [repair]")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--show-trace", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result JSON instead of the summary")
    ap.add_argument("--out", default="runs", help="directory for run folders (report.html, result.json)")
    a = ap.parse_args(argv)

    source = load_source(a.input)
    extractor = ReplayExtractor(a.replay) if a.replay else make_extractor(a.model, a.temperature, a.reasoning_effort, a.max_output_tokens)
    result = run_pipeline(source, extractor, repair=not a.no_repair)
    run_dir = save_run(result, Path(a.input.rstrip("/")).stem or "recipe", Path(a.out))
    data = result_json(result)
    if a.json:
        print(json.dumps(data, indent=1, default=str, ensure_ascii=False))
        return 0

    v, s, u = result.final.validation, data["summary"], data["usage"]
    print(f"\n{(v.recipe.title if v.recipe else a.input)}")
    print(f"status: {s['status']}   repair: {result.repair_decision}")
    cost = f"  ~${u['cost_usd']}" if u["cost_usd"] is not None else ""
    print(f"calls: {u['calls']}  tokens: {u['input_tokens']} in / {u['output_tokens']} out  time: {u['latency_s']}s{cost}")
    for at in result.attempts:
        if at.call.error:
            print(f"  {at.kind} call error: {at.call.error[:300]}")
    print(f"quotes verified: {s['grounding_rate']:.0%}   instruction coverage: {s['instruction_coverage']:.0%}")
    if v.recipe:
        print("\nSTEPS")
        for st in v.recipe.steps:
            ins = ", ".join(i.ref + ("(req)" if i.mode == "require" else "") for i in st.inputs)
            q = st.evidence.quote or f"[{st.evidence.prov}: {st.evidence.reason}]"
            print(f"  {st.id:>4} {st.verb + ('/' + st.style if st.style else ''):18} {ins:42} -> {st.output or '':16} {q!r}"
                  + ("  (optional)" if st.optional else ""))
    for title, key in (("EXTRACTION ERRORS", "extraction_errors"), ("RECIPE FINDINGS", "recipe_findings"),
                       ("UNRESOLVED", "unresolved"), ("WARNINGS", "warnings")):
        if s[key]:
            print(f"\n{title}")
            for x in s[key]:
                print(f"  {x['code']:24} {x['step'] or '':6} {x['message']}")
    if data["coverage_gaps"]:
        print("\nUNCITED INSTRUCTION TEXT")
        for g in data["coverage_gaps"]:
            print(f"  {g.strip()!r}")
    if v.edges:
        print("\nDEPENDENCIES  " + "  ".join(f"{x}->{y}" for x, y in v.edges))
    if a.show_trace and v.engine:
        print("\nSTATE TRACE")
        for t in v.engine.trace:
            if "equipment" in t:
                print(f"  {t['step']:>4} {t['verb']:10} equipment {t['equipment']}")
            elif t["status"] == "ok":
                print(f"  {t['step']:>4} {t['verb']:10} {t['consumed']} -> {t['output']} @ {t['location']} "
                      f"cooked={t['cooked_components']} props={t['properties']}")
            else:
                print(f"  {t['step']:>4} {t['verb']:10} {t['status']} {t.get('violations', '')}")
    print(f"\nreport: {run_dir / 'report.html'}")
    return 0 if s["status"] != "extraction_errors" else 1


if __name__ == "__main__":
    sys.exit(main())
