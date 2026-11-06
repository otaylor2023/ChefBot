"""Run tasks several times in the symbolic kitchen and tabulate the results.

    python -m chefbot.scripts.eval_agent                                 # every job in chefbot/jobs, 1 run each
    python -m chefbot.scripts.eval_agent --tasks fried_egg breakfast --runs 3 --jobs 4
    python -m chefbot.scripts.eval_agent --summarize runs/eval_2026-10-15_1800   # re-read an eval's runs

Each run is its own process (run_agent --quiet). Seeds differ per run, which
changes injected failures. Writes runs/eval_<stamp>/results.json and prints a
table: success by ground truth, outcome, turns, sim time, cost, check failures,
and why it failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "chefbot" / "jobs"


def run_one(task: Path, seed: int, out: Path, extra: list[str]) -> dict:
    cmd = [sys.executable, "-m", "chefbot.scripts.run_agent", "--task", str(task), "--seed", str(seed),
           "--out", str(out), "--quiet", "--operator", "abort", *extra]  # fmt: skip
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        result = json.loads(last)
    except json.JSONDecodeError:
        result = {"outcome": "crashed", "success": False, "error": (proc.stderr or proc.stdout)[-1500:]}
    return {"task": task.stem, "seed": seed, **result}


def table(results: list[dict]) -> str:
    rows = [f"{'task':<22}{'seed':>5}  {'gt':<5}{'outcome':<16}{'turns':>6}{'sim':>7}{'cost':>8}{'chk✗':>6}  why"]
    for r in sorted(results, key=lambda r: (r["task"], r["seed"])):
        t = r.get("totals") or {}
        why = "; ".join(r.get("missing", []) + [f"forbidden {p}" for p in r.get("forbidden_present", [])])
        if r.get("error"):
            why = r["error"].strip().splitlines()[-1][:80]
        rows.append(
            f"{r['task']:<22}{r['seed']:>5}  {'✓' if r.get('success') else '✗':<5}{r.get('outcome', '?'):<16}"
            f"{t.get('turns', 0):>6}{_clock(t.get('sim_s', 0)):>7}{t.get('cost_usd', 0):>8.2f}{t.get('check_failures', 0):>6}  {why[:90]}"
        )
    by_task: dict[str, list] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    rows.append("")
    for task, rs in sorted(by_task.items()):
        ok = sum(bool(r.get("success")) for r in rs)
        cost = sum((r.get("totals") or {}).get("cost_usd", 0) for r in rs)
        turns = sum((r.get("totals") or {}).get("turns", 0) for r in rs) / len(rs)
        rows.append(f"{task:<22} {ok}/{len(rs)} succeeded · {turns:.0f} turns on average · ${cost:.2f}")
    total = sum((r.get("totals") or {}).get("cost_usd", 0) for r in results)
    rows.append(f"{'all':<22} {sum(bool(r.get('success')) for r in results)}/{len(results)} · ${total:.2f}")
    return "\n".join(rows)


def _clock(t: float) -> str:
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the agent on tasks in the symbolic kitchen.")
    parser.add_argument("--tasks", nargs="*", help="task names (default: every file in chefbot/jobs)")
    parser.add_argument("--runs", type=int, default=1, help="runs per task (seeds 0..runs-1)")
    parser.add_argument("--jobs", type=int, default=4, help="runs in parallel")
    parser.add_argument("--model")
    parser.add_argument("--agent", help="the agent (chefbot/agents/<name>; run_agent's default if not given)")
    parser.add_argument("--interface", help="the session's robot API (run_agent's default if not given)")
    parser.add_argument("--plan-from", choices=["job", "prompt"], default="job",
                        help="job: each task's hand-written plan; prompt: the planner plans from each task's prompt:")
    parser.add_argument("--no-user", action="store_true", help="nobody answers the planner's questions (an ablation)")
    parser.add_argument("--label", default="", help="a name for this eval's folder (runs/eval_<stamp>_<label>)")
    parser.add_argument("--window", type=int, help="the executor's rolling window (calls kept verbatim; agent default if not given)")
    parser.add_argument("--max-cost", type=float, default=3.0, help="per-run LLM cost limit (USD)")
    parser.add_argument("--summarize", type=Path, help="print the table of an earlier eval and exit")
    parser.add_argument("--noise", choices=["off", "task", "on"], default="task",
                        help="random failures: off (clean), task (each task's rates), on (noise.yaml under them)")  # fmt: skip
    args = parser.parse_args()
    if args.summarize:
        print(table(json.loads((args.summarize / "results.json").read_text())))
        return 0
    names = args.tasks or sorted(p.stem for p in TASKS.glob("*.yaml"))
    out = REPO / "runs" / f"eval_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_noise-{args.noise}{'_' + args.label if args.label else ''}"
    out.mkdir(parents=True, exist_ok=True)
    extra = ["--max-cost", str(args.max_cost), "--noise", args.noise] + (["--model", args.model] if args.model else [])
    extra += ["--agent", args.agent] if args.agent else []
    extra += ["--interface", args.interface] if args.interface else []
    extra += ["--plan-from", args.plan_from] + (["--no-user"] if args.no_user else [])
    extra += ["--window", str(args.window)] if args.window else []
    jobs = [(TASKS / f"{n}.yaml", seed) for n in names for seed in range(args.runs)]
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(lambda job: run_one(job[0], job[1], out, extra), jobs):
            results.append(result)
            print(f"  {result['task']} s{result['seed']}: {'✓' if result.get('success') else '✗'} {result.get('outcome')}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print()
    print(table(results))
    print(f"\nruns in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
