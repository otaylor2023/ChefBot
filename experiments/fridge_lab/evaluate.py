"""The final evaluation: a frozen skill on the test split (RoboCasa's test fridges,
which the researcher never ran), plus the oracle's ceiling on the same episodes.

    python -m experiments.fridge_lab.evaluate --skill ~/fridge_lab/workspace/skills/final [--split test] [--model-usd 3]
    python -m experiments.fridge_lab.evaluate --oracle [--split test]

Writes $FRIDGE_LAB_ROOT/eval/<name>/{results.json, summary.md} and the episodes' videos.
The skill is copied first (the copy is what runs); the model ledger is separate from the research run's.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from experiments.astranomy.host import REPO
from experiments.fridge_lab import splits
from experiments.fridge_lab.lab_server import LAB_ROOT, _world_readable, _worker_init, _worker_run, snapshot_dir, summarize
from experiments.ortho_grasp.query.models import load_env


def _oracle_run(scene: dict, seed: int, out: str) -> dict:
    from experiments.fridge_lab.kitchen import LabHost
    from experiments.fridge_lab.oracle import run

    host = LabHost(scene["layout"], scene["style"], Path(out), record=False)
    try:
        t0 = time.time()
        r = run(host, scene["door"], seed)
        final = host.door_deg()
        return {"scene": scene["id"], "seed": seed, "final_deg": round(final, 2), "max_deg": round(r["max_deg"], 2), "success60": final >= 60,
                "success85": final >= 85, "sim_s": round(r["steps"] / 15, 2), "robot_time_s": round(r["steps"] / 15, 2), "think_s": 0.0,
                "model_s": 0.0, "model_usd": 0.0, "wall_s": round(time.time() - t0, 2)}  # fmt: skip
    finally:
        host.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", type=Path)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val", "dev"])
    ap.add_argument("--seeds", default=None, help="comma list (default: the split's fixed seeds; dev: 0,1)")
    ap.add_argument("--model-usd", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--vision-url", default=os.environ.get("FRIDGE_VISION_URL"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--scenes", default=None, help="comma list of scene ids (default: the whole split)")
    args = ap.parse_args()
    if not args.oracle and not args.skill:
        ap.error("--skill or --oracle")
    load_env(REPO / ".env")
    sp = splits.load()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(
        sp["test_seeds"] if args.split == "test" else sp["val_seeds"] if args.split == "val" else (0, 1))  # fmt: skip
    scenes = [s for s in sp["scenes"] if s["split"] == args.split and (not args.scenes or s["id"] in args.scenes.split(","))]
    name = args.name or (f"oracle_{args.split}" if args.oracle else f"{args.skill.name}_{args.split}_{time.strftime('%m%d-%H%M')}")
    root = LAB_ROOT / "eval" / name
    root.mkdir(parents=True, exist_ok=True)
    episodes = [(s, sd) for s in scenes for sd in seeds]
    results = []
    ctx = mp.get_context("spawn")
    env = {k: v for k, v in os.environ.items() if k.startswith(("MUJOCO", "PYOPENGL", "GEMINI", "FRIDGE", "EGL", "CUDA"))}
    if not args.oracle:
        skill = snapshot_dir() / f"eval-{name}"
        if not skill.exists():
            shutil.copytree(args.skill, skill, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            _world_readable(skill)
        ledger = (str(root / "model_ledger.jsonl"), args.model_usd)
    # Rerunning with the same --name resumes: finished episodes are read back. A worker that dies (a native crash in the
    # renderer) breaks the whole pool; the unfinished episodes then go to a fresh pool, and after 3 crashes count as failures.
    crashes: dict[tuple, int] = {}
    todo = []
    for s, sd in episodes:
        f = root / f"{s['id']}_s{sd}" / "result.json"
        if f.exists():
            results.append(json.loads(f.read_text()))
        else:
            todo.append((s, sd))
    while todo:
        with ProcessPoolExecutor(args.jobs, mp_context=ctx, initializer=_worker_init, initargs=(env,)) as ex:
            if args.oracle:
                futs = {ex.submit(_oracle_run, s, sd, str(root / f"{s['id']}_s{sd}")): (s, sd) for s, sd in todo}
            else:
                futs = {ex.submit(_worker_run, s, sd, str(skill), str(root / f"{s['id']}_s{sd}"), ledger, args.vision_url, not args.no_video,
                                  f"eval/{name}/{s['id']}_s{sd}", None): (s, sd) for s, sd in todo}  # fmt: skip
            todo = []
            for f in as_completed(futs):
                s, sd = futs[f]
                try:
                    r = f.result()
                except BrokenProcessPool:
                    key = (s["id"], sd)
                    crashes[key] = crashes.get(key, 0) + 1
                    if crashes[key] < 3:
                        todo.append((s, sd))
                        continue
                    r = {"scene": s["id"], "seed": sd, "final_deg": None, "success60": False, "success85": False, "error": "worker crashed 3 times"}
                results.append(r)
                print(f"{r['scene']} s{r['seed']}: {r.get('final_deg')} deg  {'OK' if r.get('success60') else 'fail'}  "
                      f"robot {r.get('robot_time_s')} s  ${r.get('model_usd')}  {r.get('error') or ''}", flush=True)  # fmt: skip
        if todo:
            print(f"worker pool broke; rerunning {len(todo)} unfinished episodes", flush=True)
    results.sort(key=lambda r: (r["scene"], r["seed"]))
    summ = summarize(results)
    by_type: dict[str, list] = {}
    types = {s["id"]: s["type"] for s in scenes}
    for r in results:
        by_type.setdefault(types[r["scene"]], []).append(r)
    out = {"name": name, "split": args.split, "seeds": seeds, "skill": None if args.oracle else str(args.skill), "summary": summ,
           "by_type": {t: summarize(v) for t, v in by_type.items()}, "episodes": results}  # fmt: skip
    (root / "results.json").write_text(json.dumps(out, indent=1))
    lines = [f"# {name}", "", f"split {args.split}, {len(scenes)} scenes x seeds {seeds}", "",
             "| | success60 | success85 | robot time (s) | model $ |", "|---|---|---|---|---|",
             f"| all | {summ['success60']:.2f} | {summ['success85']:.2f} | {summ['mean_robot_time_s']} | {summ['mean_model_usd']} |"]  # fmt: skip
    for t, v in sorted(out["by_type"].items()):
        lines.append(f"| {t} | {v['success60']:.2f} | {v['success85']:.2f} | {v['mean_robot_time_s']} | {v['mean_model_usd']} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
