"""Metrics over eval runs, and before/after comparisons (the evaluation's tracking).

    python -m chefbot.scripts.eval_metrics runs/eval_A runs/eval_B ...          # a row per eval
    python -m chefbot.scripts.eval_metrics --compare runs/eval_A runs/eval_B    # paired by task and seed
    python -m chefbot.scripts.eval_metrics --tasks find_mug_moved ... runs/eval_A runs/eval_B

Per run, from its run.jsonl: success (ground truth), turns, LLM cost (agent + planner), errors, repeated failures
(the same failing call made again: within 8 calls, or later), complete refused by the check, complete blocked, the
longest run of complete calls failing in a row, finding calls (detect/go_to before the first pick_up of the run), and
the planner's questions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WINDOW = 8


def run_metrics(run_dir: Path) -> dict:
    lines = [json.loads(l) for l in (run_dir / "run.jsonl").read_text().splitlines() if l.strip()]
    acts = [a for a in lines if a.get("kind") == "action"]
    seen: dict = {}
    rep_in = rep_out = 0
    complete_streak = longest = refused = blocked = 0
    first_pick = next((i for i, a in enumerate(acts) if a["tool"] == "pick_up" and a["result"]["success"]), len(acts))
    for i, a in enumerate(acts):
        key = (a["tool"], json.dumps(a.get("args", {}), sort_keys=True))
        ok = a["result"]["success"]
        if key in seen and not seen[key][1] and not ok:
            rep_in += i - seen[key][0] <= WINDOW
            rep_out += i - seen[key][0] > WINDOW
        seen[key] = (i, ok)
        if a["tool"] == "complete":
            err = a["result"].get("error")
            refused += err == "check_failed"
            blocked += err == "complete_refused_again"
            complete_streak = complete_streak + 1 if not ok else 0
            longest = max(longest, complete_streak)
        elif a["tool"] not in ("set_status",):
            complete_streak = 0
    finding = sum(a["tool"] in ("detect", "go_to", "nearby", "open") for a in acts[:first_pick])
    planner = [l for l in lines if l.get("kind") == "planner"]
    return {"calls": len(acts), "errors": sum(not a["result"]["success"] for a in acts), "repeat_fail_in_window": rep_in,
            "repeat_fail_after_window": rep_out, "complete_refused": refused, "complete_blocked": blocked,
            "complete_loop": longest, "finding_calls": finding, "questions": sum(len(p.get("questions") or []) for p in planner),
            "planner_rejected": sum(p.get("rejected", 0) for p in planner)}  # fmt: skip


def load(eval_dir: Path, tasks: set[str] | None = None) -> list[dict]:
    rows = []
    for r in json.loads((eval_dir / "results.json").read_text()):
        if tasks and r["task"] not in tasks:
            continue
        t = r.get("totals") or {}
        m = run_metrics(Path(r["run_dir"])) if r.get("run_dir") and (Path(r["run_dir"]) / "run.jsonl").exists() else {}
        rows.append({"task": r["task"], "seed": r["seed"], "success": bool(r.get("success")), "outcome": r.get("outcome"),
                     "turns": t.get("turns", 0), "cost": t.get("cost_usd", 0.0), "sim_s": t.get("sim_s", 0.0), **m})  # fmt: skip
    return rows


def summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    avg = lambda k: sum(r.get(k, 0) for r in rows) / n  # noqa: E731
    return {"runs": len(rows), "success": sum(r["success"] for r in rows), "success_rate": sum(r["success"] for r in rows) / n,
            "turns": avg("turns"), "cost": sum(r["cost"] for r in rows), "errors": avg("errors"),
            "repeat_fail_in_window": avg("repeat_fail_in_window"), "repeat_fail_after_window": avg("repeat_fail_after_window"),
            "complete_refused": avg("complete_refused"), "complete_loop_max": max((r.get("complete_loop", 0) for r in rows), default=0),
            "finding_calls": avg("finding_calls"), "questions": avg("questions")}  # fmt: skip


def table(evals: list[tuple[str, list[dict]]]) -> str:
    head = f"{'eval':<44}{'runs':>5}{'succ':>8}{'turns':>7}{'cost$':>8}{'err':>6}{'rep<=8':>7}{'rep>8':>6}{'refuse':>7}{'loop':>5}{'find':>6}{'ask':>5}"
    out = [head]
    for name, rows in evals:
        s = summary(rows)
        out.append(f"{name[-44:]:<44}{s['runs']:>5}{s['success']:>4}/{s['runs']:<3}{s['turns']:>7.1f}{s['cost']:>8.2f}{s['errors']:>6.1f}"
                   f"{s['repeat_fail_in_window']:>7.2f}{s['repeat_fail_after_window']:>6.2f}{s['complete_refused']:>7.2f}"
                   f"{s['complete_loop_max']:>5}{s['finding_calls']:>6.1f}{s['questions']:>5.2f}")  # fmt: skip
    return "\n".join(out)


def compare(a: list[dict], b: list[dict]) -> str:
    key = lambda r: (r["task"], r["seed"])  # noqa: E731
    bb = {key(r): r for r in b}
    out = [f"{'task':<28}{'seed':>5}  {'before':<14}{'after':<14}{'turns':>12}{'cost$':>14}"]
    for r in sorted(a, key=key):
        o = bb.get(key(r))
        if o is None:
            continue
        mark = lambda x: ("✓ " if x["success"] else "✗ ") + str(x["outcome"])[:11]  # noqa: E731
        out.append(f"{r['task']:<28}{r['seed']:>5}  {mark(r):<14}{mark(o):<14}{r['turns']:>5}->{o['turns']:<6}{r['cost']:>6.2f}->{o['cost']:<6.2f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+", type=Path)
    ap.add_argument("--compare", action="store_true", help="paired before/after of the first two evals")
    ap.add_argument("--tasks", nargs="*", help="only these tasks")
    ap.add_argument("--json", type=Path, help="write the summaries here")
    a = ap.parse_args()
    tasks = set(a.tasks) if a.tasks else None
    evals = [(str(e), load(e, tasks)) for e in a.evals]
    print(table(evals))
    if a.compare and len(evals) >= 2:
        print()
        print(compare(evals[0][1], evals[1][1]))
    if a.json:
        a.json.write_text(json.dumps({name: {"summary": summary(rows), "runs": rows} for name, rows in evals}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
