"""Memory across runs (after MessyMem's long sequence without a reset): task A then task B in one kitchen, B run
with the memory A left (run_agent --memory) and without it (the env's map, from before A). Writes results.json like
eval_agent, the tasks named <B>+memory, <B>-memory and <A>.

    python -m chefbot.scripts.eval_sequences --runs 3 [--plan-from prompt]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from chefbot.scripts.eval_agent import REPO, TASKS, run_one, table

SEQUENCES = [("seq_mug_A", "seq_mug_B"), ("seq_plate_A", "seq_plate_B"), ("seq_egg_A", "seq_egg_B")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--plan-from", choices=["job", "prompt"], default="job")
    ap.add_argument("--label", default="sequences")
    a = ap.parse_args()
    out = REPO / "runs" / f"eval_{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    extra = ["--max-cost", "3.0", "--noise", "task", "--plan-from", a.plan_from]

    def one(item):
        (first, second), seed = item
        memory = out / f"{second}_s{seed}_memory.yaml"
        rows = [{**run_one(TASKS / f"{first}.yaml", seed, out, extra + ["--memory", str(memory)]), "task": first}]
        rows.append({**run_one(TASKS / f"{second}.yaml", seed, out, extra + ["--memory", str(memory)]), "task": f"{second}+memory"})
        rows.append({**run_one(TASKS / f"{second}.yaml", seed, out, extra), "task": f"{second}-memory"})
        return rows

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for rows in pool.map(one, [(s, seed) for s in SEQUENCES for seed in range(a.runs)]):
            results += rows
            print("  " + "  ".join(f"{r['task']} s{r['seed']}: {'✓' if r.get('success') else '✗'}" for r in rows), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print()
    print(table(results))
    print(f"\nruns in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
