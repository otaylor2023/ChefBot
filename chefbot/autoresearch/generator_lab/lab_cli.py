#!/usr/bin/env python3
"""./lab: the researcher's interface to the object lab (standard library only).

  ./lab build GEN [--seeds 0,1,2 | 0-7] [--no-video] [--wait]
                              build assets with generator GEN (generate, physics checks, renders): prints a job id
                              (or, with --wait, the results). Results: runs/<job>/seed_<n>/ (the asset; lab/ holds
                              checks.json, sheet.jpg, actuation.jpg, actuation.mp4), runs/<job>/batch.jpg
  ./lab wait JOB [--timeout 900]   wait for a build; prints each asset's checks and a summary
  ./lab status JOB | ./lab jobs     progress
  ./lab judge JOB [--seeds 0,3]    the dev judge on a finished build's assets (and the batch's variety): scores,
                              flaws. Each asset is one judge call; the batch one more (budgeted: ./lab budget)
  ./lab val GEN                    GEN on fresh hidden seeds, judged against held-out references by a stronger
                              judge: aggregate only; limited submissions. Prints a job id; then ./lab wait JOB
  ./lab budget                     judge calls left, val submissions left, time left

GEN is a directory under generators/ with a generate.py and a ranges.yaml, e.g. generators/v3.
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


def seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part.strip()[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def show_assets(assets: list[dict]) -> None:
    for a in assets:
        m = a.get("measured") or {}
        nums = "  ".join(f"{k}={v}" for k, v in m.items())
        print(f"seed {a['seed']:>4}  {'PASS' if a.get('ok') else 'FAIL'}  gen {a.get('gen_s')} s  {nums}")
        for k, why in (a.get("why") or {}).items():
            print(f"            {k}: {why}")
        for k in ("harness_error", "render_error"):
            if a.get(k):
                print(f"            {k}: {a[k]}")


def show(r: dict) -> None:
    if isinstance(r.get("assets"), list) and r["assets"] and "ok" in r["assets"][0]:
        show_assets(r["assets"])
        r = {k: v for k, v in r.items() if k != "assets"}
    elif isinstance(r.get("assets"), list):  # judge results
        for a in r["assets"]:
            if "error" in a:
                print(f"seed {a['seed']:>4}  judge error: {a['error']}")
                continue
            scores = "  ".join(f"{k}={a[k]}" for k in ("overall", "shape", "tips", "mechanism", "materials", "proportions"))
            print(f"seed {a['seed']:>4}  {scores}  ({a.get('closest_real', '')})")
            print(f"            {a.get('summary', '')}")
            for f in a.get("flaws", []):
                print(f"            - {f}")
        r = {k: v for k, v in r.items() if k != "assets"}
    print(json.dumps(r, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(prog="./lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "wait", "status", "jobs", "judge", "val", "budget"])
    ap.add_argument("arg", nargs="?", help="GEN (build/val) or JOB (wait/status/judge)")
    ap.add_argument("--seeds", default=None, help="build: default 0; judge: default all")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    if a.cmd in ("build", "val") and not a.arg:
        ap.error(f"{a.cmd} needs a generator directory (e.g. generators/v1)")
    if a.cmd in ("wait", "status", "judge") and not a.arg:
        ap.error(f"{a.cmd} needs a JOB id")
    if a.cmd == "build":
        req = {"cmd": "build", "generator": a.arg, "seeds": seeds(a.seeds or "0"), "no_video": a.no_video}
        if a.wait:
            req["wait_s"] = a.timeout
        r = call(req, (a.timeout if a.wait else 0) + 120)
    elif a.cmd == "val":
        r = call({"cmd": "val", "generator": a.arg}, 120)
    elif a.cmd in ("wait", "status"):
        r = call({"cmd": "status", "job": a.arg, "wait_s": a.timeout if a.cmd == "wait" else 0}, (a.timeout if a.cmd == "wait" else 0) + 60)
    elif a.cmd == "judge":
        r = call({"cmd": "judge", "job": a.arg, "seeds": seeds(a.seeds) if a.seeds else []}, 1800)
    else:
        r = call({"cmd": a.cmd}, 60)
    show(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
