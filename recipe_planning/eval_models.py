#!/usr/bin/env python
"""Compare models on one recipe sample: who succeeds, how accurate, how action-ready, at what cost; which recipes
are hard. Runs eval_llm.py once per model, then joins the results.

  python eval_models.py --recipes ../../recipe_corpus --subset robot_plausible --n 12 \
      --models ollama:qwen3.5:9b ollama:gpt-oss:20b openrouter:openai/gpt-6.1-sol --reference openrouter:openai/gpt-6.1-sol
  python eval_models.py --recipes recipes --models ollama:gemma4:e4b-it-q8_0     # the 3 gold recipes
  python eval_models.py --join runs/models__20261105-2000                        # rebuild the tables only

Accuracy has two references:
  gold    recipes/gold/<name>.gold.json (human-checked) when it exists: eval_llm's gold scores.
  silver  --reference MODEL's first run on the same recipe. Agreement with a strong model, NOT correctness:
          use it to rank models and find hard recipes, and promote reviewed silver outputs to gold.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from recipe_ir.adapters import from_text
from recipe_ir.compare import compare
from recipe_ir.corpus import load_index, select
from recipe_ir.recipe_pipeline import validate
from recipe_ir.schema import Recipe

HERE = Path(__file__).resolve().parent


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(st.mean(xs)), 3) if xs else None


def load_runs(eval_dir: Path) -> dict[str, list[dict]]:
    """recipe -> [{run, result.json fields, actions.json, run_dir}] for one model's eval folder."""
    out: dict[str, list[dict]] = {}
    data = json.loads((eval_dir / "eval.json").read_text())
    for name, rec in data["recipes"].items():
        for r in rec["runs"]:
            if "run_dir" not in r:
                out.setdefault(name, []).append({**r, "api_error": True})
                continue
            d = Path(r["run_dir"])
            if not d.exists():
                d = next((c for c in (HERE / d, eval_dir / "runs" / d.name) if c.exists()), d)
            res = json.loads((d / "result.json").read_text())
            act = json.loads((d / "actions.json").read_text()) if (d / "actions.json").exists() else None
            out.setdefault(name, []).append({**r, "dir": d, "result": res, "actions": act})
    return out


def revalidate(run: dict):
    """The saved recipe, re-validated against the exact source the model saw (fills evidence spans)."""
    if not run.get("result") or not run["result"].get("recipe"):
        return None
    src = from_text((run["dir"] / "source.txt").read_text(encoding="utf-8"))
    return validate(Recipe.model_validate(run["result"]["recipe"]), src)


def join(batch: Path, reference: str | None, recipes_dir: Path | None):
    manifest = json.loads((batch / "manifest.json").read_text())
    models = manifest["models"]
    runs = {m: load_runs(Path(d)) for m, d in manifest["eval_dirs"].items() if d}
    index = {r["name"]: r for r in load_index(recipes_dir)} if recipes_dir and (recipes_dir / "index.jsonl").exists() else {}
    names = sorted({n for m in runs.values() for n in m})
    ref_v = {}
    if reference and reference in runs:
        for n in names:
            first = next((r for r in runs[reference].get(n, []) if not r.get("api_error")), None)
            v = revalidate(first) if first else None
            if v is not None and v.recipe is not None and v.status != "extraction_errors":
                ref_v[n] = v

    cells, per_model = {}, {}
    for m in models:
        rows = []
        for n in names:
            rs = runs.get(m, {}).get(n, [])
            ok = [r for r in rs if not r.get("api_error")]
            silver = []
            for r in ok:
                if m == reference or n not in ref_v:
                    continue
                v = revalidate(r)
                if v is not None and v.recipe is not None:
                    c = compare(v.recipe, ref_v[n].recipe, set(v.edges), set(ref_v[n].edges))
                    silver.append(c)
            feas = [r["actions"]["feasibility"] for r in ok if r.get("actions") and r["actions"].get("feasibility")]
            ins = [r["actions"]["instructions"] for r in ok if r.get("actions")]
            cell = {
                "runs": len(rs), "api_errors": len(rs) - len(ok),
                "valid": mean([r["status_final"] != "extraction_errors" for r in ok]),
                "valid_first": mean([r["status_first"] != "extraction_errors" for r in ok]),
                "grounding": mean([r["grounding"] for r in ok]), "coverage": mean([r["coverage"] for r in ok]),
                "steps": mean([r["n_steps"] for r in ok]),
                "gold_strict_step_f1": mean([r["gold"]["strict_step_f1"] for r in ok if "gold" in r]),
                "gold_edge_f1": mean([r["gold"]["edge_f1"] for r in ok if "gold" in r]),
                "silver_strict_step_f1": mean([c.strict_step_f1 for c in silver]),
                "silver_edge_f1": mean([c.edge_f1 for c in silver]),
                "open_questions": mean([i["counts"]["open_questions"] for i in ins if "counts" in i]),
                "robot_supported_share": mean([f["supported_share"] for f in feas]),
                "cost_usd": mean([r.get("cost_usd") for r in ok]), "latency_s": mean([r.get("latency_s") for r in ok]),
                "tokens_out": mean([r.get("output_tokens") for r in ok]),
            }
            cells[(m, n)] = cell
            rows.append(cell)
        per_model[m] = {k: mean([c[k] for c in rows]) for k in rows[0]} if rows else {}
        if rows:
            per_model[m]["total_cost_usd"] = round(sum((c["cost_usd"] or 0) * c["runs"] for c in rows), 4)

    difficulty = []
    for n in names:
        cs = [cells[(m, n)] for m in models if (m, n) in cells]
        f = index.get(n, {})
        difficulty.append({"recipe": n, "site": f.get("site"), "size": f.get("size"),
                           "robot_plausible": f.get("robot_plausible"),
                           "models_valid": mean([c["valid"] for c in cs]),
                           "silver_step_f1": mean([c["silver_strict_step_f1"] for c in cs]),
                           "gold_step_f1": mean([c["gold_strict_step_f1"] for c in cs]),
                           "has_reference": n in ref_v, **{k: f.get(k) for k in ("n_sentences", "oven", "dough")}})
    difficulty.sort(key=lambda d: (d["models_valid"] if d["models_valid"] is not None else 9,
                                   d["silver_step_f1"] if d["silver_step_f1"] is not None else 9))

    cols = ["valid_first", "valid", "grounding", "coverage", "gold_strict_step_f1", "gold_edge_f1",
            "silver_strict_step_f1", "silver_edge_f1", "open_questions", "robot_supported_share", "cost_usd",
            "latency_s", "total_cost_usd"]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = [f"# Model comparison\n\n{len(names)} recipes, models: {', '.join(models)}; silver reference: {reference or 'none'}\n",
          "| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    md += ["| " + m + " | " + " | ".join(fmt(per_model.get(m, {}).get(c)) for c in cols) + " |" for m in models]
    md += ["\n## Valid (after repair) per recipe\n", "| recipe | " + " | ".join(models) + " |", "|" + "---|" * (len(models) + 1)]
    md += ["| " + n + " | " + " | ".join(fmt(cells.get((m, n), {}).get("valid")) for m in models) + " |" for n in names]
    md += ["\n## Hardest recipes first\n", "| recipe | site | size | robot_plausible | models_valid | silver_step_f1 | gold_step_f1 |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {d['recipe']} | {d['site']} | {d['size']} | {d['robot_plausible']} | {fmt(d['models_valid'])} | "
           f"{fmt(d['silver_step_f1'])} | {fmt(d['gold_step_f1'])} |" for d in difficulty]
    md.append("\nsilver = agreement with the reference model, not correctness. gold = human-checked reference.")
    (batch / "matrix.md").write_text("\n".join(md) + "\n")
    (batch / "matrix.json").write_text(json.dumps({"models": per_model, "difficulty": difficulty,
                                                   "cells": {f"{m}||{n}": c for (m, n), c in cells.items()}}, indent=1))
    with (batch / "cells.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(next(iter(cells.values())).keys()) if cells else []
        w.writerow(["model", "recipe", *keys])
        for (m, n), c in cells.items():
            w.writerow([m, n, *[c[k] for k in keys]])
    print("\n".join(md[:len(models) + 3]))
    print(f"\nsaved: {batch}/matrix.md, matrix.json, cells.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--recipes", default="recipes")
    ap.add_argument("--subset", default="all", choices=["all", "robot_plausible", "broad"])
    ap.add_argument("--n", type=int, help="sample size (stratified by size); default all")
    ap.add_argument("--only", nargs="*", help="explicit recipe names (overrides --subset/--n)")
    ap.add_argument("--max-per-site", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--reference", help="model whose first valid run is the silver reference")
    ap.add_argument("--reasoning-effort", help="passed to every model (default: eval_llm's default for OpenAI; none for others)")
    ap.add_argument("--max-cost", type=float, default=2.0, help="per model, recorded spend (USD); local models cost 0")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--join", metavar="BATCH_DIR", help="only rebuild tables for an existing batch")
    a = ap.parse_args(argv)
    rdir = Path(a.recipes).resolve()
    if a.join:
        join(Path(a.join), a.reference or json.loads((Path(a.join) / "manifest.json").read_text()).get("reference"), rdir)
        return 0
    if not a.models:
        ap.error("--models is required unless --join")
    if a.only:
        names = a.only
    elif (rdir / "index.jsonl").exists() or any(rdir.glob("*.meta.json")):
        names = select(load_index(rdir), subset=a.subset, n=a.n, seed=a.seed, max_per_site=a.max_per_site)
    else:
        names = sorted(p.stem for p in rdir.glob("*.txt"))[: a.n]
    if not names:
        ap.error("no recipes selected")
    batch = Path(a.out).resolve() / f"models__{time.strftime('%Y%m%d-%H%M%S')}"
    batch.mkdir(parents=True, exist_ok=True)
    manifest = {"models": a.models, "reference": a.reference, "recipes": names, "recipes_dir": str(rdir.resolve()),
                "subset": a.subset, "seed": a.seed, "runs": a.runs, "eval_dirs": {}}
    print(f"{len(names)} recipes x {len(a.models)} models x {a.runs} run(s) -> {batch}", file=sys.stderr)
    order = ([a.reference] if a.reference in a.models else []) + [m for m in a.models if m != a.reference]
    for m in order:
        before = set((batch / "evals").glob("eval__*")) if (batch / "evals").exists() else set()
        cmd = [sys.executable, str(HERE / "eval_llm.py"), "--model", m, "--recipes", str(rdir), "--runs", str(a.runs),
               "--out", str(batch / "evals"), "--max-cost", str(a.max_cost), "--only", *names]
        eff = a.reasoning_effort if a.reasoning_effort is not None else ("" if not m.startswith(("gpt", "o", "openai:")) else None)
        if eff is not None:
            cmd += ["--reasoning-effort", eff]
        print(f"\n=== {m}", file=sys.stderr, flush=True)
        rc = subprocess.call(cmd, cwd=HERE)
        new = sorted(set((batch / "evals").glob("eval__*")) - before)
        manifest["eval_dirs"][m] = str(new[-1].resolve()) if new and (new[-1] / "eval.json").exists() else None
        if rc != 0 or manifest["eval_dirs"][m] is None:
            print(f"  {m}: eval_llm exited {rc}; see output above", file=sys.stderr)
        (batch / "manifest.json").write_text(json.dumps(manifest, indent=1))
    join(batch, a.reference, rdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
