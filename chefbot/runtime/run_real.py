"""Run a lab skill on the real robot: the same sandboxed episode as in the sims (runtime/episode.py), its host the
robot (environments/real/host.py).

    python -m chefbot.runtime.run_real --skill pick_up/chefbot_v30 --arg object="red cup" [--dry-run] [--yes]

Before: the robot up with the onboard server (onboard/bringup/launch.sh: the arm and base drivers, OpenSai, the
server), and on this computer `ssh -L 8765:localhost:8765 tidybot01` (environments/real/config.yaml's url), the
vision server for skills that use it (VISION_URL), the model keys in .env.

--home first takes the arm to its ready pose (the hand pointing down). --confirm: Enter before each move (the default runs them; Ctrl-C stops). --dry-run: nothing is sent to the arm or the gripper, the arm taken
to be where it was commanded, the cameras real. Arm only unless --base. The episode's files (result.json,
trace.jsonl, calls.jsonl, skill_log.txt, the skill's images) go to runs/real/<skill>__<time>/; nothing scores it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from chefbot import embodiments, skills
from chefbot.runtime import models as M
from chefbot.runtime.episode import LIMITS, run_episode

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "real"


def _arg(s: str) -> tuple[str, object]:
    k, _, v = s.partition("=")
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main() -> None:
    from chefbot.environments.real.host import FLANGE_Z_MIN, RealHost

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", required=True, help="a registered skill, e.g. pick_up/chefbot_v30")
    ap.add_argument("--arg", action="append", default=[], type=_arg, help="a skill argument, name=value (repeatable)")
    ap.add_argument("--robot", default="tidybot_fr3", help="the embodiment the real robot is (default tidybot_fr3)")
    ap.add_argument("--speed", default="slow", help="speed profile (default slow: first real runs)")
    ap.add_argument("--dry-run", action="store_true", help="send nothing to the arm or the gripper")
    ap.add_argument("--confirm", action="store_true", help="wait for Enter before each move")
    ap.add_argument("--base", action="store_true", help="let the base move (default: it holds)")
    ap.add_argument("--home", action="store_true", help="the arm to its ready pose before the skill (a skill can do it itself: act_joints)")
    ap.add_argument("--flange-z-min", type=float, default=FLANGE_Z_MIN, help="the flange's height floor over the base, arm base frame (m)")
    ap.add_argument("--url", help="the onboard server (default: environments/real/config.yaml's)")
    ap.add_argument("--model-usd", type=float, default=LIMITS["model_usd"], help="the episode's model budget")
    ap.add_argument("--think-s", type=float, default=LIMITS["think_s"])
    ap.add_argument("--steps", type=int, default=LIMITS["control_steps"], help="control steps (15 Hz)")
    a = ap.parse_args()

    M.load_env(REPO / ".env")
    skill = skills.load(a.skill)
    robot = embodiments.load(a.robot)
    vision_url = os.environ.get("VISION_URL")
    try:
        skill.check(robot.name, vision_url)
    except skills.MissingRequirement as e:
        if f"built for robot {skill.robot}" not in str(e) or "needs" in str(e):
            raise SystemExit(str(e))
        print(f"[run_real] note: {skill.path} was built for robot {skill.robot}; running it on {robot.name} (the real robot)")
    access = {**skill.access(), "confirm_moves": a.confirm}
    if not a.base and not access.get("arm_only"):
        print("[run_real] note: the skill isn't arm only, but the base holds (--base lets it move)")
        access["arm_only"] = True
    out = RUNS / f"{skill.path.replace('/', '-')}__{time.strftime('%m%d-%H%M%S')}{'_dry' if a.dry_run else ''}"
    host = RealHost(robot, robot.speed(a.speed), out, dry_run=a.dry_run, confirm=a.confirm, base=a.base, url=a.url,
                    flange_z_min=a.flange_z_min)  # fmt: skip
    if a.home:
        try:
            host.go_home()
        except BaseException:
            host.close()
            raise
    ledger = M.Ledger(out / "model_ledger.jsonl", a.model_usd)
    vision = M.Vision(vision_url) if access.get("vision") else None
    limits = {**LIMITS, "model_usd": a.model_usd, "think_s": a.think_s, "control_steps": a.steps}
    print(f"[run_real] {skill.path} on the real robot ({robot.name}, speed {a.speed}) with {dict(a.arg)}; out {out}")
    try:
        result = run_episode(host, None, 0, skill.dir, out, ledger, vision, access, interface=skill.interface, limits=limits,
                             record=False, tag="real", args=dict(a.arg) or None)  # fmt: skip
    finally:
        host.close()
    print(json.dumps({k: result[k] for k in ("end", "returned", "steps", "robot_time_s", "think_s", "model_usd", "n_llm", "n_vision", "n_observe")}, indent=1))
    print(f"[run_real] files in {out}")


if __name__ == "__main__":
    main()
