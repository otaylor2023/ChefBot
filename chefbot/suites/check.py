"""Check a suite set before a lab uses it: every item's setup builds, and its task finds a clear start at each of its
seeds (no skill runs). An item that can't start would be a harness error in every episode of it.

    python -m chefbot.suites.check robocasa_pick [--robot panda_franka] [--jobs 4]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chefbot import suites


def check_item(setup_name: str, task_name: str, seeds: list[int], robot_name: str) -> dict:
    from chefbot import embodiments, setups
    from chefbot.environments import make_host

    t0, host = time.time(), None
    try:
        robot = embodiments.load(robot_name)
        setup = setups.load(setup_name)
        task = setup.task(task_name)
        host = make_host(setup, robot, robot.speed(), Path(tempfile.mkdtemp(prefix="chefbot_check_")), record=False)
        failed = []
        for seed in seeds:
            try:
                host.start(task, seed)
            except RuntimeError as e:
                failed.append((seed, str(e)))
        return {"setup": setup_name, "failed": failed, "s": round(time.time() - t0, 1)}
    except Exception:  # noqa: BLE001
        return {"setup": setup_name, "error": traceback.format_exc().splitlines()[-1], "s": round(time.time() - t0, 1)}
    finally:
        if host is not None:
            host.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="a suite set (robocasa_pick) or one suite (robocasa_pick/dev)")
    ap.add_argument("--robot", default="panda_franka")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    parts = suites.load_set(a.suite) if suites.is_set(a.suite) else {a.suite: suites.load(a.suite)}
    items = [(part, it) for part, s in parts.items() for it in s.items]
    bad = 0
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as pool:
        futs = [(part, it, pool.submit(check_item, it.setup, it.task, it.seeds, a.robot)) for part, it in items]
        for part, it, f in futs:
            r = f.result()
            problem = r.get("error") or "; ".join(f"seed {s}: {e}" for s, e in r["failed"])
            bad += bool(problem)
            print(f"{part:5s} {it.id:32s} {'FAIL ' + problem if problem else 'ok'}  ({r['s']} s)", flush=True)
    print(f"{len(items) - bad}/{len(items)} items start at all their seeds")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
