#!/usr/bin/env python3
"""./lab: the researcher's interface to the fridge lab (standard library only).

  ./lab scenes                                   the dev scenes (id, fridge type, model, door)
  ./lab run SKILL --scene dev03 [--seed 0]       one dev episode; waits; prints its result and folder
  ./lab suite SKILL [--scenes all|dev01,dev04] [--seeds 0,1] [--no-video]
                                                 a batch of dev episodes in the background; prints a job id
  ./lab wait JOB [--timeout 900]                 wait for a job; prints per-episode results and a summary
  ./lab status JOB | ./lab jobs                  progress
  ./lab val SKILL                                score SKILL on the held-out val scenes (aggregate only;
                                                 limited submissions per night): prints a job id; then ./lab wait
  ./lab budget                                   model dollars spent, val submissions left, time left
  ./lab profile --prompt TEXT [--image PNG] [--effort low] [--tier default] [--n 1] [--reps 3]
                                                 (Astra lab) time GPT-6 Astra calls, no robot: latency per call

SKILL is a directory under skills/ with a main.py, e.g. skills/v3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

IPC = Path(__file__).resolve().parent / ".lab" / "ipc"


def call(req: dict, timeout: float) -> dict:
    rid = uuid.uuid4().hex[:12]
    tmp = IPC / f".req_{rid}.json"
    tmp.write_text(json.dumps(req))
    tmp.rename(IPC / f"req_{rid}.json")
    resp = IPC / f"resp_{rid}.json"
    t_end = time.time() + timeout
    while time.time() < t_end:
        if resp.exists():
            out = json.loads(resp.read_text())
            resp.unlink()
            return out
        time.sleep(0.3)
    return {"error": f"no answer from the lab in {timeout:.0f} s (is the lab server running?)"}


def table(eps: list[dict]) -> str:
    cols = ["scene", "seed", "final_deg", "success60", "success85", "robot_time_s", "sim_s", "think_s", "model_s", "model_usd", "end"]
    if any("n_astra" in e for e in eps):
        cols = ["scene", "seed", "final_deg", "success60", "success85", "robot_time_s", "think_s", "n_astra", "astra_s", "end"]
    lines = ["  ".join(f"{c:>12s}" if c != "end" else c for c in cols)]
    for e in eps:
        if e.get("error"):
            lines.append(f"{e.get('scene')!s:>12s}  {e.get('seed')!s:>12s}  HARNESS ERROR: {e['error']}")
            continue
        lines.append("  ".join(f"{str(e.get(c)):>12s}" if c != "end" else str(e.get(c)) for c in cols))
    return "\n".join(lines)


def show(r: dict) -> None:
    if "episodes" in r and isinstance(r["episodes"], list):
        print(table(r["episodes"]))
        r = {k: v for k, v in r.items() if k != "episodes"}
    print(json.dumps(r, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(prog="./lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scenes", "run", "suite", "wait", "status", "jobs", "val", "budget", "profile"])
    ap.add_argument("arg", nargs="?", help="SKILL (run/suite/val) or JOB (wait/status)")
    ap.add_argument("--scene")
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--prompt")
    ap.add_argument("--image", action="append", default=[], help="a PNG/JPG in the workspace (repeatable)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tier", default="default")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.cmd in ("run", "suite", "val") and not a.arg:
        ap.error(f"{a.cmd} needs a SKILL directory (e.g. skills/v1)")
    if a.cmd in ("wait", "status") and not a.arg:
        ap.error(f"{a.cmd} needs a JOB id")
    if a.cmd == "scenes":
        r = call({"cmd": "scenes"}, 60)
        for s in r.get("scenes", []):
            print(f"{s['id']}  {s['type']:22s} {s['model']}  door {s['door']}")
        if "error" in r:
            print(r["error"])
        return 0
    if a.cmd == "run":
        if not a.scene:
            ap.error("run needs --scene")
        show(call({"cmd": "run", "skill": a.arg, "scenes": [a.scene], "seeds": [a.seed], "no_video": a.no_video, "wait_s": a.timeout}, a.timeout + 60))
        return 0
    if a.cmd == "suite":
        r = call({"cmd": "suite", "skill": a.arg, "scenes": a.scenes.split(","), "seeds": [int(s) for s in a.seeds.split(",")],
                  "no_video": a.no_video}, 120)  # fmt: skip
    elif a.cmd == "profile":
        if not a.prompt:
            ap.error("profile needs --prompt")
        imgs = [str(Path(i).resolve()) for i in a.image]
        r = call({"cmd": "profile", "prompt": a.prompt, "images": imgs, "effort": a.effort, "tier": a.tier, "n": a.n, "reps": a.reps},
                 a.reps * 150 + 60)  # fmt: skip
    elif a.cmd == "val":
        r = call({"cmd": "val", "skill": a.arg}, 120)
    elif a.cmd in ("wait", "status"):
        r = call({"cmd": "status", "job": a.arg, "wait_s": a.timeout if a.cmd == "wait" else 0}, (a.timeout if a.cmd == "wait" else 0) + 60)
    else:
        r = call({"cmd": a.cmd}, 60)
    show(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
