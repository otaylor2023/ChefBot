"""One episode in a worker process. A worker keeps its last host (kitchen) and reuses it while items share a setup.

`skill` describes what runs: {"dir", "interface", "access" (model calls it may make)}; for a registered skill,
skills.Skill.spec() gives it; a lab builds one for a researcher's skill from its config.
(agentic-skills-exploration's evals/worker.py.)
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

_host = None  # (setup name, robot, speed, camera noise) -> host


def init(env: dict) -> None:
    os.environ.update(env)


def run(item: dict, seed: int, skill: dict, robot_name: str, speed_name: str | None, out: str, ledger: tuple,
        vision_url: str | None, record: bool, tag: str, limits: dict | None, camera_noise: bool = True) -> dict:  # fmt: skip
    global _host
    from chefbot.runtime import models as M
    from chefbot import embodiments, setups
    from chefbot.runtime.episode import run_episode
    from chefbot.environments.robocasa.host import make_host

    try:
        robot = embodiments.load(robot_name)
        setup = setups.load(item["setup"])
        task = setup.task(item["task"])
        key = (item["setup"], robot_name, speed_name, camera_noise)
        if _host is None or _host[0] != key:
            if _host is not None:
                _host[1].close()
            _host = (key, make_host(setup, robot, robot.speed(speed_name), Path(out), record=record, camera_noise=camera_noise))
        host = _host[1]
        return run_episode(host, task, seed, Path(skill["dir"]), Path(out), M.Ledger(Path(ledger[0]), ledger[1]), M.Vision(vision_url),
                           skill["access"], skill.get("interface", "chefbot-v1"), limits=limits, record=record, tag=tag, item=item["id"])  # fmt: skip
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()[-1500:]
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "harness_error.txt").write_text(err)
        if _host is not None:  # a broken host is rebuilt next time
            try:
                _host[1].close()
            except Exception:  # noqa: BLE001
                pass
            _host = None
        return {"item": item["id"], "setup": item["setup"], "task": item["task"], "seed": seed, "error": err.splitlines()[-1], "success": False}
