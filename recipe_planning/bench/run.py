#!/usr/bin/env python
"""Run one model (and one agent configuration) on RecipeBench instances.

  python -m bench.run --instances ../../recipe_eval_v1/bench/instances_lite.jsonl --model ollama:gpt-oss:20b
  python -m bench.run --instances ... --model ollama:qwen3.5:9b --reasoning-effort off --tag no-repair --no-repair
  RECIPE_REPAIR_HINTS=0 python -m bench.run --instances ... --model ... --tag no-hints

Configurations (the agent loop under test): repair+hints (default), no-hints (validator findings only), no-repair
(one extraction call, no feedback loop). Results go to <out>/<model>__<tag>/results.jsonl, one row per instance x
run. Rerunning the same command resumes: finished (instance, run) pairs are skipped.

A MISTAKE is a first attempt with extraction errors. RECOVERY is a mistake whose final output has none.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipe_ir.adapters import from_text  # noqa: E402
from recipe_ir.config import DEFAULT_REASONING_EFFORT  # noqa: E402
from recipe_ir.llm_extractor import make_extractor, PROMPT_VERSION  # noqa: E402
from recipe_ir.recipe_pipeline import run_pipeline, validate  # noqa: E402
from recipe_ir.runio import run_cost, save_run  # noqa: E402
from recipe_ir.schema import Recipe  # noqa: E402
from bench.checks import check  # noqa: E402


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None, help="configuration label (default: repair+hints / no-repair / no-hints)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--max-output-tokens", type=int, default=8000)
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--only", nargs="*", help="perturbation names to include (base, reorder, noise, ...)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-cost", type=float, help="stop starting new instances once recorded spend reaches this (USD)")
    ap.add_argument("--out", default="bench_runs")
    a = ap.parse_args(argv)
    if a.reasoning_effort is None:
        a.reasoning_effort = DEFAULT_REASONING_EFFORT if a.model.startswith(("gpt", "o", "openai:")) else ""
    hints = os.environ.get("RECIPE_REPAIR_HINTS", "1") != "0"
    tag = a.tag or ("no-repair" if a.no_repair else ("repair+hints" if hints else "no-hints"))
    if a.temperature:
        tag += f"@t{a.temperature:g}"
    inst_path = Path(a.instances)
    refs = inst_path.parent / "references"
    instances = [json.loads(l) for l in inst_path.read_text().splitlines() if l.strip()]
    if a.only:
        instances = [i for i in instances if i["perturbation"] in a.only]
    instances = instances[: a.limit] if a.limit else instances
    out = Path(a.out) / f"{slug(a.model)}__{slug(tag)}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "tag": tag, "instances": str(inst_path.resolve()), "split": inst_path.stem,
            "prompt_version": PROMPT_VERSION, "temperature": a.temperature, "reasoning_effort": a.reasoning_effort,
            "repair": not a.no_repair, "repair_hints": hints}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    results_path = out / "results.jsonl"
    done = set()
    spent = 0.0
    if results_path.exists():
        for l in results_path.read_text().splitlines():
            if l.strip():
                row = json.loads(l)
                done.add((row["id"], row["run"]))
                spent += row.get("cost_usd") or 0.0
    ref_cache: dict[str, object] = {}

    def reference(name):
        if name not in ref_cache:
            rj = json.loads((refs / f"{name}.json").read_text())
            ref_cache[name] = validate(Recipe.model_validate(rj["recipe"]), from_text(rj["source"]))
        return ref_cache[name]

    lock = threading.Lock()
    stop = {"why": None}

    def one(inst, k):
        if stop["why"]:
            return None
        src = from_text(inst["text"])
        ex = make_extractor(a.model, a.temperature, a.reasoning_effort, a.max_output_tokens)
        t0 = time.perf_counter()
        try:
            res = run_pipeline(src, ex, repair=not a.no_repair)
        except Exception as e:  # an API / server failure is a data point
            return {"id": inst["id"], "run": k, "base": inst["base"], "perturbation": inst["perturbation"],
                    "split": inst["split"], "model": a.model, "tag": tag, "resolved": False,
                    "reasons": [f"call failed: {type(e).__name__}: {str(e)[:200]}"], "api_error": True,
                    "latency_s": round(time.perf_counter() - t0, 1), "cost_usd": None}
        v, first = res.final.validation, res.attempts[0].validation
        verdict = check(inst, v, reference(inst["reference"]))
        mistake = bool(first.extraction_errors)
        u = res.usage_totals()
        run_dir = save_run(res, f"{inst['id']}-r{k}", out / "runs")
        return {"id": inst["id"], "run": k, "base": inst["base"], "perturbation": inst["perturbation"],
                "split": inst["split"], "model": a.model, "tag": tag, **verdict,
                "first_status": first.status, "final_status": v.status,
                "first_errors": sorted({x.code for x in first.extraction_errors}),
                "final_errors": sorted({x.code for x in v.extraction_errors}),
                "mistake": mistake, "recovered": mistake and not v.extraction_errors,
                "repair": res.repair_decision, "calls": u["calls"], "input_tokens": u["input_tokens"],
                "output_tokens": u["output_tokens"], "latency_s": u["latency_s"], "cost_usd": run_cost(res),
                "truncated": sum(1 for at in res.attempts if (at.call.error or "").startswith("truncated")),
                "adaptations": sorted(getattr(ex, "adaptations", set())),
                "run_dir": str(run_dir)}

    jobs = [(i, k) for k in range(a.runs) for i in instances if (i["id"], k) not in done]
    workers = a.workers or (1 if a.model.startswith("ollama:") else 6)
    print(f"{a.model} [{tag}]: {len(jobs)} to run, {len(done)} already done -> {out}", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, results_path.open("a") as fh:
        futs = [pool.submit(one, i, k) for i, k in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            row = f.result()
            if row is None:
                continue
            with lock:
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                spent += row.get("cost_usd") or 0.0
                if a.max_cost is not None and spent >= a.max_cost and not stop["why"]:
                    stop["why"] = f"--max-cost {a.max_cost} reached (${spent:.2f})"
                    print("  " + stop["why"], file=sys.stderr)
            mark = "PASS" if row["resolved"] else "fail"
            print(f"  [{n}/{len(jobs)}] {mark} {row['id']} r{row['run']} ({row.get('latency_s')}s)"
                  + ("" if row["resolved"] else f"  {'; '.join(row['reasons'])[:160]}"), file=sys.stderr, flush=True)
    print(f"done; recorded spend ${spent:.2f}" + (f"; {stop['why']}" if stop["why"] else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
