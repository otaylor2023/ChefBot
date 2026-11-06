#!/usr/bin/env python
"""Evaluate LLM extraction over a folder of recipes.

For every recipes/<name>.txt (or .html), runs the pipeline N times and reports:
  validity before/after repair, repair acceptance, quote grounding, instruction coverage, unmodeled verbs,
  gold agreement (if recipes/gold/<name>.gold.json exists): step / verb / edge / ingredient scores, recipe-finding
  agreement, repeatability (pairwise agreement between runs), and tokens / latency / cost.

  python eval_llm.py --model gpt-6.1-sol --reasoning-effort low --runs 3
  python eval_llm.py --gold-as-prediction        # offline sanity check: every gold score must be 1.0
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from recipe_ir.config import DEFAULT_RECIPE_MODEL, DEFAULT_REASONING_EFFORT
from recipe_ir.adapters import load_source
from recipe_ir.compare import compare
from recipe_ir.schema import Recipe
from recipe_ir.llm_extractor import make_extractor, ReplayExtractor, PROMPT_VERSION
from recipe_ir.recipe_pipeline import run_pipeline, validate
from recipe_ir.runio import save_run, run_cost


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(st.mean(xs)), 3) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_RECIPE_MODEL)
    ap.add_argument("--recipes", default="recipes")
    ap.add_argument("--only", nargs="*", help="recipe names to include")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0, help="dropped automatically if the model rejects it")
    ap.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, help="OpenAI reasoning models only, e.g. low / medium / high")
    ap.add_argument("--max-output-tokens", type=int, default=8000, help="caps the cost of a runaway response")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--gold-as-prediction", action="store_true", help="offline: feed gold back as the model output")
    ap.add_argument("--rescore", metavar="EVAL_DIR", help="offline: re-validate the model outputs saved in an earlier eval "
                    "folder with the current validator and gold files (no API calls; tokens/cost are the recorded ones)")
    ap.add_argument("--workers", type=int, default=None, help="parallel extractions (default 6; 1 for ollama:*)")
    ap.add_argument("--limit", type=int, help="only the first N recipes (alphabetical): a cheap smoke test")
    ap.add_argument("--max-cost", type=float, help="stop starting new runs once recorded spend (USD) reaches this")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)

    if a.workers is None:
        a.workers = 1 if a.model.startswith("ollama:") else 6
    rdir = Path(a.recipes)
    files = sorted(p for p in rdir.iterdir() if p.suffix in (".txt", ".html", ".htm"))
    if a.limit:
        files = files[:a.limit]
    if a.only:
        files = [p for p in files if p.stem in a.only]
    recorded: dict[tuple[str, int], dict] = {}
    if a.rescore:
        import re as _re
        for rj in sorted(Path(a.rescore).glob("runs/*/result.json")):
            m = _re.match(r"(.+)-r(\d+)__(.+?)__", rj.parent.name)
            if m:
                recorded[(m.group(1), int(m.group(2)))] = {**json.loads(rj.read_text()), "_dir": rj.parent}
        a.model = next(iter(recorded.values()))["usage"]["model"] if recorded else a.model
        # Always the source.txt each run saved: exactly what the model saw (a recipe file of the same name in
        # --recipes may be a different recipe). Gold is still looked up in --recipes/gold.
        src_dir = Path(a.out) / f"_rescore_sources_{time.strftime('%Y%m%d-%H%M%S')}"
        files = []
        for stem in sorted({k[0] for k in recorded}):
            src = next(v["_dir"] / "source.txt" for k, v in recorded.items() if k[0] == stem)
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / f"{stem}.txt").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            files.append(src_dir / f"{stem}.txt")
    label = "gold" if a.gold_as_prediction else (f"rescore-{a.model}" if a.rescore else a.model)
    out_dir = Path(a.out) / f"eval__{label}__{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_recipe = [], {}

    def one_run(f, source, gold_path, gold_v, i):
        rec_old = recorded.get((f.stem, i))
        if rec_old is not None:
            tmp = out_dir / "replay" / f"{f.stem}-r{i}"
            tmp.mkdir(parents=True, exist_ok=True)
            paths = []
            for k, att in enumerate(rec_old["attempts"]):
                (tmp / f"{k}.json").write_text(att["raw"])
                paths.append(tmp / f"{k}.json")
            ex = ReplayExtractor(paths)
        elif a.gold_as_prediction:
            ex = ReplayExtractor([gold_path])
        else:
            ex = make_extractor(a.model, a.temperature, a.reasoning_effort, a.max_output_tokens)
        try:
            res = run_pipeline(source, ex, repair=not a.no_repair)
        except Exception as e:  # an API failure is a data point, not a crash
            print(f"  {f.stem} run {i}: ERROR {e}", file=sys.stderr)
            return {"run": i, "error": f"{type(e).__name__}: {e}"}
        first, final = res.attempts[0].validation, res.final.validation
        u = res.usage_totals()
        if rec_old is not None:  # keep what the real calls cost, not the replay's zeros
            u = {k: rec_old["usage"][k] for k in ("input_tokens", "output_tokens", "latency_s", "calls")}
        rec = {
            "run": i, "status_first": first.status, "status_final": final.status,
            "errors_first": len(first.extraction_errors), "errors_final": len(final.extraction_errors),
            "error_codes_final": sorted({x.code for x in final.extraction_errors}),
            "repair": res.repair_decision,
            "truncated_calls": (sum(1 for at in rec_old["attempts"] if (at.get("error") or "").startswith("truncated"))
                                if rec_old is not None else
                                sum(1 for at in res.attempts if (at.call.error or "").startswith("truncated"))), "grounding": round(final.grounding_rate(), 3),
            "coverage": round(final.coverage, 3), "n_steps": len(final.recipe.steps) if final.recipe else 0,
            "n_rules": len(final.recipe.rules) if final.recipe else 0,
            "uncited_actions": len(final.uncited(source.text)["action"]) if final.recipe else None,
            "uncited_rules": len(final.uncited(source.text)["rule"]) if final.recipe else None,
            "unmodeled_verbs": sum(1 for v in (final.engine.unresolved if final.engine else []) if v.code == "unmodeled_verb"),
            "unresolved": len(final.engine.unresolved) if final.engine else 0,
            **u, "cost_usd": (rec_old["usage"].get("cost_usd") if rec_old is not None else run_cost(res)),
            "run_dir": str(save_run(res, f"{f.stem}-r{i}", out_dir / "runs")),
        }
        if gold_v is not None and final.recipe is not None and gold_v.recipe is not None:
            rec["gold"] = compare(final.recipe, gold_v.recipe, set(final.edges), set(gold_v.edges)).to_dict()
            rec["finding_agrees"] = {x.code for x in final.recipe_findings} == {x.code for x in gold_v.recipe_findings}
        rec["_v"] = final
        print(f"  {f.stem} run {i}: {final.status} ({u['latency_s']}s)", file=sys.stderr)
        return rec

    jobs, ctx = [], {}
    for f in files:
        source = load_source(str(f))
        gold_path = rdir / "gold" / f"{f.stem}.gold.json"
        if gold_path.exists() and a.rescore and load_source(str(rdir / f"{f.stem}.txt")).text != source.text:
            gold_path = rdir / "gold" / "__none__"   # same name, different recipe: no gold
        gold_v = validate(Recipe.model_validate_json(gold_path.read_text()), source) if gold_path.exists() else None
        ctx[f.stem] = (f, source, gold_path, gold_v)
        if a.gold_as_prediction and not gold_path.exists():
            continue
        if a.rescore:
            jobs += [(f.stem, i) for (stem, i) in sorted(recorded) if stem == f.stem]
        else:
            jobs += [(f.stem, i) for i in range(a.runs)]
    t0 = time.perf_counter()
    results: dict[str, list] = {k: [] for k in ctx}
    spent, stopped, processed = 0.0, None, set()
    pool = ThreadPoolExecutor(max_workers=max(1, a.workers))
    futs = {pool.submit(one_run, *ctx[name], i): name for name, i in jobs}
    try:
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            rec = fut.result()
            processed.add(fut)
            results[futs[fut]].append(rec)
            spent += rec.get("cost_usd") or 0.0
            if a.max_cost is not None and spent >= a.max_cost and stopped is None:
                stopped = f"--max-cost {a.max_cost} reached (${spent:.2f})"
                for f in futs:
                    f.cancel()                      # queued runs never start; running ones finish
                print(f"  {stopped}: not starting new runs; waiting for {a.workers} in flight at most", file=sys.stderr)
    except KeyboardInterrupt:
        stopped = "interrupted"
        for f in futs:
            f.cancel()
        print("\n  interrupted: summarising finished runs; calls still in flight are abandoned", file=sys.stderr)
        for fut in futs:
            if fut.done() and not fut.cancelled() and fut not in processed:
                results[futs[fut]].append(fut.result())
    finally:
        pool.shutdown(wait=stopped != "interrupted", cancel_futures=True)
    wall = round(time.perf_counter() - t0, 1)
    ctx = {k: v for k, v in ctx.items() if results[k]}   # recipes with at least one finished run

    for name, (f, source, gold_path, gold_v) in ctx.items():
        runs = sorted(results[name], key=lambda r: r["run"])
        ok = [r for r in runs if "error" not in r]
        pairs = [compare(x["_v"].recipe, y["_v"].recipe, set(x["_v"].edges), set(y["_v"].edges))
                 for x, y in itertools.combinations(ok, 2) if x["_v"].recipe and y["_v"].recipe]
        g = [r["gold"] for r in ok if "gold" in r]
        summary = {
            "runs": len(runs), "api_errors": len(runs) - len(ok),
            "valid_first": mean([r["status_first"] != "extraction_errors" for r in ok]),
            "valid_final": mean([r["status_final"] != "extraction_errors" for r in ok]),
            "repairs_accepted": sum(r["repair"].startswith("accepted") for r in ok),
            "repairs_attempted": sum(r["repair"].startswith(("accepted", "rejected")) for r in ok),
            "grounding": mean([r["grounding"] for r in ok]), "coverage": mean([r["coverage"] for r in ok]),
            "uncited_actions": mean([r.get("uncited_actions") for r in ok]),
            "uncited_rules": mean([r.get("uncited_rules") for r in ok]),
            "rules": mean([r.get("n_rules") for r in ok]), "steps": mean([r.get("n_steps") for r in ok]),
            "unmodeled_verbs": mean([r["unmodeled_verbs"] for r in ok]),
            "step_f1": mean([x["step_f1"] for x in g]), "strict_step_f1": mean([x["strict_step_f1"] for x in g]),
            "truncated_calls": sum(r.get("truncated_calls", 0) for r in ok), "verb_acc": mean([x["verb_acc"] for x in g]),
            "edge_f1": mean([x["edge_f1"] for x in g]), "ingredient_f1": mean([x["ingredient_f1"] for x in g]),
            "needs_cooking_acc": mean([x["needs_cooking_acc"] for x in g]),
            "finding_agrees": mean([r.get("finding_agrees") for r in ok if "finding_agrees" in r]),
            "repeat_step_f1": mean([p.step_f1 for p in pairs]), "repeat_edge_f1": mean([p.edge_f1 for p in pairs]),
            "tokens_in": mean([r["input_tokens"] for r in ok]), "tokens_out": mean([r["output_tokens"] for r in ok]),
            "latency_s": mean([r["latency_s"] for r in ok]), "cost_usd": mean([r["cost_usd"] for r in ok]),
            "error_codes": sorted({c for r in ok for c in r["error_codes_final"]}),
            "missed_steps": sorted({m for x in g for m in x["missed_steps"]}),
            "verb_errors": sorted({m for x in g for m in x["verb_errors"]}),
        }
        per_recipe[f.stem] = {"summary": summary, "runs": [{k: v for k, v in r.items() if k != "_v"} for r in runs]}
        rows.append((f.stem, summary))

    cols = ["valid_first", "valid_final", "grounding", "steps", "uncited_actions", "rules", "uncited_rules",
            "strict_step_f1", "edge_f1", "finding_agrees", "repeat_edge_f1", "truncated_calls", "tokens_out",
            "latency_s", "cost_usd"]
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    header = "| recipe | " + " | ".join(cols) + " |"
    lines = [header, "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + name + " | " + " | ".join(fmt(s[c]) for c in cols) + " |" for name, s in rows]
    overall = {c: mean([s[c] for _, s in rows]) for c in cols}
    lines.append("| **mean** | " + " | ".join(fmt(overall[c]) for c in cols) + " |")
    table = "\n".join(lines)
    meta = {"model": "gold" if a.gold_as_prediction else a.model, "prompt_version": PROMPT_VERSION,
            "runs_per_recipe": a.runs, "repair": not a.no_repair, "temperature": a.temperature,
            "reasoning_effort": a.reasoning_effort, "workers": a.workers, "wall_clock_s": wall,
            "spent_usd": round(spent, 4), "stopped": stopped}
    (out_dir / "eval.json").write_text(json.dumps({"meta": meta, "overall": overall, "recipes": per_recipe}, indent=1, default=str))
    notes = "\n".join(f"- **{n}** remaining errors: {', '.join(s['error_codes']) or 'none'}; missed: "
                      f"{', '.join(s['missed_steps']) or 'none'}; verb errors: {', '.join(s['verb_errors']) or 'none'}"
                      for n, s in rows)
    (out_dir / "eval.md").write_text(f"# Extraction eval\n\n`{json.dumps(meta)}`\n\n{table}\n\n{notes}\n")
    print(table)
    print(f"\n{notes}\n\nwall clock: {wall}s with {a.workers} workers; recorded spend ${spent:.2f}"
          + (f"; STOPPED EARLY: {stopped}" if stopped else "")
          + f"\nsaved: {out_dir}/eval.md, eval.json, runs/*/report.html")
    if stopped == "interrupted":
        sys.stdout.flush()
        os._exit(130)   # don't wait for abandoned HTTP calls in worker threads
    return 0


if __name__ == "__main__":
    sys.exit(main())
