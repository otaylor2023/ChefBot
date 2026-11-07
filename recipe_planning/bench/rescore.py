#!/usr/bin/env python
"""Re-apply the current checks to saved results, offline (no model calls). Use after fixing a check.

  python -m bench.rescore --runs ../../recipe_eval_v1/bench_runs --instances-dir ../../recipe_eval_v1/bench

Each results.jsonl is backed up to results.jsonl.bak-<time> before rows are rewritten. Only the verdict fields
(resolved, reasons, metrics) change; what the model did (statuses, mistakes, recovery, cost) stays as recorded.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from recipe_ir.adapters import from_text  # noqa: E402
from recipe_ir.recipe_pipeline import validate  # noqa: E402
from recipe_ir.schema import Recipe  # noqa: E402
from bench.checks import check  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--instances-dir", required=True)
    a = ap.parse_args(argv)
    idir = Path(a.instances_dir)
    instances = {}
    for f in idir.glob("instances_*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                i = json.loads(l)
                instances[i["id"]] = i
    refs = {}

    def ref(name):
        if name not in refs:
            rj = json.loads((idir / "references" / f"{name}.json").read_text())
            refs[name] = validate(Recipe.model_validate(rj["recipe"]), from_text(rj["source"]))
        return refs[name]

    for rp in sorted(Path(a.runs).glob("*/results.jsonl")):
        rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
        changed = 0
        for r in rows:
            d = Path(r.get("run_dir", ""))
            if r.get("api_error") or not (d / "result.json").exists():
                continue
            res = json.loads((d / "result.json").read_text())
            if not res.get("recipe"):
                continue
            v = validate(Recipe.model_validate(res["recipe"]), from_text((d / "source.txt").read_text(encoding="utf-8")))
            verdict = check(instances[r["id"]], v, ref(instances[r["id"]]["reference"]))
            if verdict["resolved"] != r["resolved"] or verdict["reasons"] != r["reasons"]:
                changed += 1
            r.update(verdict)
        shutil.copy2(rp, rp.with_name(f"results.jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
        rp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(f"{rp.parent.name}: {changed} verdicts changed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
