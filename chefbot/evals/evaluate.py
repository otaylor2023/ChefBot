"""Run an eval: one skill on a suite or a suite set, with a robot. Results go to runs/evals/<id>/ and a line in
runs/evals/index.jsonl. The test suites are run only this way, after a lab.

    python -m chefbot.evals.evaluate --skill pick_up/run1_v12 --suite robocasa_pick            # a set: dev, val and test
    python -m chefbot.evals.evaluate --skill pick_up/run1_v12 --suite robocasa_pick/test --jobs 4 --group-by object_types
    python -m chefbot.evals.evaluate --skill pick_up/run1_v12 --suite robocasa_pick/dev --items dev_00_apricot --seeds 0 --video

runs/evals/<id>/: eval.yaml (what ran, with every version involved), <suite part>/<item>_s<seed>/ (result.json,
video, trace, logs), results.json, summary.json, summary.md.
(agentic-skills-exploration's evals/run.py.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from chefbot import embodiments, skills, suites

from chefbot.runtime import worker
from chefbot.runtime.episode import LIMITS
from chefbot.autoresearch.lab_server import WORKER_ENV
from chefbot.runtime.models import load_env
from chefbot.autoresearch.research import REPO, RUNS
from chefbot.autoresearch.summary import grouped, summarize, table


def _digest(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        for f in sorted([p] if p.is_file() else (x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)):
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def stamp(skill, suite_names: list[str], robot, speed: str) -> dict:
    """Every version that shaped the results: the code, the skill, the suites, the robot and its speed profile."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"code": git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else ""),
            "skill": _digest([skill.dir / "main.py", skill.dir / "skill.yaml"]), "interface": skill.interface,
            "suites": {n: _digest([suites.ROOT / f"{n}.yaml"]) for n in suite_names},
            "robot": _digest([robot.dir]), "speed": speed}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", required=True, help="a registered skill, e.g. pick_up/run1_v12")
    ap.add_argument("--suite", required=True, help="a suite set (robocasa_pick) or one suite (robocasa_pick/test)")
    ap.add_argument("--robot", help="default: the skill's")
    ap.add_argument("--speed", help="speed profile (default: the robot's)")
    ap.add_argument("--items", help="only these item ids (comma-separated)")
    ap.add_argument("--seeds", help="override the items' seeds (comma-separated)")
    ap.add_argument("--group-by", default="", help="setup attributes to break results down by (comma-separated)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--video", action="store_true", help="record video (slower)")
    ap.add_argument("--no-camera-noise", action="store_true")
    ap.add_argument("--model-usd", type=float, default=5.0, help="total cap on paid model calls for this eval")
    ap.add_argument("--name", help="eval id (default: <skill>__<suite>__<time>)")
    a = ap.parse_args()

    load_env(REPO / ".env")
    skill = skills.load(a.skill)
    if not a.robot and skill.robot == "any":
        raise SystemExit(f"skill {skill.path} runs on any robot: pass --robot")
    robot = embodiments.load(a.robot or skill.robot)
    speed = robot.speed(a.speed).name
    vision_url = os.environ.get("VISION_URL")
    skill.check(robot.name, vision_url)
    parts = suites.load_set(a.suite) if suites.is_set(a.suite) else {a.suite.split("/")[-1]: suites.load(a.suite)}
    wanted = set(a.items.split(",")) if a.items else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    eid = a.name or f"{skill.path.replace('/', '-')}__{a.suite.replace('/', '-')}__{time.strftime('%m%d-%H%M%S')}"
    out = RUNS / "evals" / eid
    out.mkdir(parents=True, exist_ok=False)
    group_by = [g for g in a.group_by.split(",") if g]
    suite_names = [s.name for s in parts.values()]
    spec = {"skill": skill.path, "suite": a.suite, "robot": robot.name, "speed": speed, "group_by": group_by,
            "items": sorted(wanted) if wanted else "all", "seeds": seeds or "suite's", "video": a.video,
            "camera_noise": not a.no_camera_noise, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": stamp(skill, suite_names, robot, speed)}  # fmt: skip
    (out / "eval.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    work = [(part, it, sd) for part, s in parts.items() for it in s.items if not wanted or it.id in wanted for sd in (seeds or it.seeds)]
    # Group by setup so each worker rebuilds a kitchen as rarely as possible.
    work.sort(key=lambda w: (w[1].setup, w[1].id, w[2]))
    ledger = (str(out / "model_ledger.jsonl"), a.model_usd)
    env = {k: v for k, v in os.environ.items() if k.startswith(WORKER_ENV)}
    results = {p: [] for p in parts}
    print(f"eval {eid}: {skill.path} on {a.suite} ({len(work)} episodes, robot {robot.name}, speed {speed})", flush=True)
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn"), initializer=worker.init, initargs=(env,)) as pool:
        futs = {}
        for part, it, sd in work:
            ep = out / part / f"{it.id}_s{sd}"
            futs[pool.submit(worker.run, {"id": it.id, "setup": it.setup, "task": it.task}, sd, skill.spec(), robot.name, speed, str(ep),
                             ledger, vision_url, a.video, eid, dict(LIMITS), not a.no_camera_noise)] = part  # fmt: skip
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            results[futs[f]].append(r)
            what = f"lifted {r.get('lift_cm')} cm" if r.get("lift_cm") is not None else ""
            print(f"[{i + 1}/{len(work)}] {futs[f]} {r['item']} s{r['seed']}: " + (f"ERROR {r['error']}" if "error" in r else
                  f"{what} {'OK' if r.get('success') else '--'} {r.get('robot_time_s')} s ${r.get('model_usd')}"), flush=True)  # fmt: skip

    summary, md = {}, f"# {eid}\n\n{skill.path} on {a.suite}, robot {robot.name}, speed {speed}\n"
    for part, rs in results.items():
        rs.sort(key=lambda r: (r["item"], r["seed"]))
        summary[part] = {"all": summarize(rs), "groups": grouped(rs, group_by)}
        md += f"\n## {part} ({parts[part].name})\n\n" + table(summary[part]["all"], summary[part]["groups"])
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(md)
    with open(RUNS / "evals" / "index.jsonl", "a") as idx:
        idx.write(json.dumps({"eval": eid, **{k: spec[k] for k in ("skill", "suite", "robot", "speed", "started", "versions")},
                              "summary": {p: s["all"] for p, s in summary.items()}}) + "\n")  # fmt: skip
    print(md)


if __name__ == "__main__":
    main()
