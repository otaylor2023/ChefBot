#!/usr/bin/env python
"""Aggregate every bench run folder into a leaderboard.

  python -m bench.leaderboard --runs bench_runs

Per (model, configuration, split): % resolved overall, per perturbation and per base split; first-try valid;
mistakes and recovery rate; pass^k / pass@k when runs > 1; cost and latency. Also instances.csv (instance x
configuration) to find what breaks every model.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PERTS = ["base", "reorder", "noise", "no_headings", "injection", "no_heat_level", "units", "planted_gap"]


def pct(xs):
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 1) if xs else None


STEP_COUNT = re.compile(r"^(\d+) steps vs reference (\d+) \(tolerance (\d+)%\)$")


def rejudge(r: dict, rows: list[dict]) -> dict:
    """Invariance variants (reorder, noise, no_headings, injection) test that the output does not change with the
    input. Their step-count condition is therefore judged against the SAME model's own base run, not the reference
    model's granularity. The reference comparison stays visible as `ref_step_count_ok`."""
    reasons = r.get("reasons") or []
    sc = [x for x in reasons if STEP_COUNT.match(x)]
    r = {**r, "ref_step_count_ok": not sc}
    if r["perturbation"] in ("base",) or not r.get("metrics"):
        return r
    base = next((b for b in rows if b["id"] == f"{r['base']}__base" and b["run"] == r["run"] and b.get("metrics")), None)
    if base is None or not base["metrics"].get("steps"):
        return r
    n, tol = r["metrics"].get("steps", 0), int(STEP_COUNT.match(sc[0]).group(3)) / 100 if sc else 0.3
    own_ok = abs(n - base["metrics"]["steps"]) <= tol * base["metrics"]["steps"]
    rest = [x for x in reasons if not STEP_COUNT.match(x)]
    if not own_ok and r["perturbation"] in ("reorder", "noise", "no_headings", "injection"):
        rest.append(f"{n} steps vs own base {base['metrics']['steps']} (tolerance {int(tol * 100)}%)")
    return {**r, "reasons": rest, "resolved": not rest, "own_base_steps": base["metrics"]["steps"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="bench_runs")
    a = ap.parse_args(argv)
    root = Path(a.runs)
    boards, cells = [], defaultdict(dict)
    for d in sorted(root.glob("*/results.jsonl")):
        meta = json.loads((d.parent / "meta.json").read_text())
        rows = [json.loads(l) for l in d.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        rows = [rejudge(r, rows) for r in rows]
        label = f"{meta['model']} [{meta['tag']}]"
        by_inst = defaultdict(list)
        for r in rows:
            by_inst[r["id"]].append(r)
        k = max(len(v) for v in by_inst.values())
        first_run = [v[0] for v in by_inst.values()]
        ok = [r for r in rows if not r.get("api_error")]
        mistakes = [r for r in ok if r.get("mistake")]
        b = {"config": label, "model": meta["model"], "tag": meta["tag"], "split": meta["split"],
             "prompt": meta["prompt_version"], "instances": len(by_inst), "runs": k,
             "resolved_pct": pct(r["resolved"] for r in first_run),
             "first_try_valid_pct": pct(r.get("first_status") not in (None, "extraction_errors") for r in ok),
             "mistakes": len(mistakes), "recovered": sum(r["recovered"] for r in mistakes),
             "recovery_pct": pct(r["recovered"] for r in mistakes),
             "api_errors": len(rows) - len(ok),
             "ref_granularity_agree_pct": pct(r["ref_step_count_ok"] for r in first_run
                                              if r["perturbation"] in ("reorder", "noise", "no_headings", "injection")),
             "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 3),
             "mean_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / len(rows), 1)}
        if k > 1:
            full = [v for v in by_inst.values() if len(v) == k]
            b[f"pass^{k}_pct"] = pct(all(r["resolved"] for r in v) for v in full)
            b[f"pass@{k}_pct"] = pct(any(r["resolved"] for r in v) for v in full)
        for p in PERTS:
            b[p] = pct(r["resolved"] for r in first_run if r["perturbation"] == p)
        for s in ("gold", "robot_plausible", "broad"):
            b[f"split_{s}"] = pct(r["resolved"] for r in first_run if r["split"] == s)
        boards.append(b)
        for iid, v in by_inst.items():
            cells[iid][label] = v[0]["resolved"]
            cells[iid]["_p"] = v[0]["perturbation"]
    if not boards:
        print("no results yet", file=sys.stderr)
        return 1
    boards.sort(key=lambda b: (b["split"], -(b["resolved_pct"] or 0)))
    fmt = lambda x: "-" if x is None else str(x)
    md = ["# RecipeBench leaderboard", "",
          "Resolved = no extraction errors AND the instance's own test passes (see bench/checks.py). Invariance",
          "variants compare step count with the same model's own base run; agreement with the reference model's step",
          "count is reported separately (ref granularity agree %).", ""]
    for split in sorted({b["split"] for b in boards}):
        bs = [b for b in boards if b["split"] == split]
        md += [f"## {split}", "", "| configuration | n | resolved % | 1st-try valid % | recovered / mistakes | "
               + " | ".join(PERTS) + " | cost $ | s / inst |", "|" + "---|" * (len(PERTS) + 6)]
        for b in bs:
            md.append(f"| {b['config']} | {b['instances']} | **{fmt(b['resolved_pct'])}** | {fmt(b['first_try_valid_pct'])} | "
                      f"{b['recovered']} / {b['mistakes']} | " + " | ".join(fmt(b[p]) for p in PERTS)
                      + f" | {b['cost_usd']} | {b['mean_latency_s']} |")
        md.append("")
    labels = [b["config"] for b in boards]
    never = [i for i, c in cells.items() if not any(c.get(l) for l in labels if l in c)]
    md += ["## Instances no configuration resolved", "", ", ".join(sorted(never)) or "none", ""]
    (root / "leaderboard.md").write_text("\n".join(md))
    (root / "leaderboard.json").write_text(json.dumps(boards, indent=1))
    with (root / "instances.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "perturbation", *labels])
        for iid in sorted(cells):
            w.writerow([iid, cells[iid]["_p"], *[{True: 1, False: 0}.get(cells[iid].get(l), "") for l in labels]])
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
