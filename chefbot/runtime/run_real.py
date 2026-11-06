"""Run a lab skill on the real robot: the same sandboxed episode as in the sims (runtime/episode.py), its host the
robot (environments/real/host.py).

    python -m chefbot.runtime.run_real --skill pick_up/chefbot_v30 --arg object="red cup" [--dry-run] [--yes]

Before: the robot up with the onboard server (onboard/bringup/launch.sh: the arm and base drivers, OpenSai, the
server), and on this computer environments/real/up.sh (the tunnel to the onboard server, environments/real/
config.yaml's url, and the vision server for skills that use it, VISION_URL), the model keys in .env.

--ready first takes the arm to the ready position (interface/workspace.py READY: looking forward and down; --home is the same). --confirm: Enter before each move (the default runs them; Ctrl-C stops). --dry-run: nothing is sent to the arm or the gripper, the arm taken
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
    ap.add_argument("--speed", default="slow", help="the speed profile for the arm and the base (default slow: first real runs)")
    ap.add_argument("--arm-speed", help="the arm's speed profile (default --speed)")
    ap.add_argument("--base-speed", help="the base's speed profile (default --speed; only with --base, and refused over the robot's base cap)")
    ap.add_argument("--dry-run", action="store_true", help="send nothing to the arm or the gripper")
    ap.add_argument("--confirm", action="store_true", help="wait for Enter before each move")
    ap.add_argument("--base", action="store_true", help="let the base move (default: it holds)")
    ap.add_argument("--ready", "--home", dest="ready", action="store_true", help="the arm to the ready position before the skill (--home: the same)")
    ap.add_argument("--flange-z-min", type=float, default=FLANGE_Z_MIN, help="the flange's height floor over the base, arm base frame (m)")
    ap.add_argument("--solver", help="the arm solver (embodiments/arm_solvers: qp, osc, legacy; default the robot's)")
    ap.add_argument("--url", help="the onboard server (default: environments/real/config.yaml's)")
    ap.add_argument("--model-usd", type=float, default=LIMITS["model_usd"], help="the episode's model budget")
    ap.add_argument("--think-s", type=float, default=LIMITS["think_s"])
    ap.add_argument("--steps", type=int, default=LIMITS["control_steps"], help="control steps (15 Hz)")
    a = ap.parse_args()

    M.load_env(REPO / ".env")
    skill = skills.load(a.skill)
    robot = embodiments.load(a.robot)
    if a.solver:
        robot.kin.solver = a.solver
    robot.kin.log = print
    print(f"[run_real] arm solver: {robot.kin.solver}")
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
    cams = access.get("cameras")
    try:  # arm only (no --base): the base's speed is 0 and its profile isn't checked
        speed = robot.speed(a.speed, arm=a.arm_speed, base=a.base_speed, use_base=a.base)
    except ValueError as e:
        raise SystemExit(f"[run_real] {e}")
    host = RealHost(robot, speed, out, dry_run=a.dry_run, confirm=a.confirm, base=a.base, url=a.url,
                    cameras=cams if isinstance(cams, list) else ([] if cams is False else None),
                    flange_z_min=a.flange_z_min)  # fmt: skip
    if a.ready:
        try:
            host.go_ready()
        except BaseException:
            host.close()
            raise
    ledger = M.Ledger(out / "model_ledger.jsonl", a.model_usd)
    vision = M.Vision(vision_url) if access.get("vision") else None
    limits = {**LIMITS, "model_usd": a.model_usd, "think_s": a.think_s, "control_steps": a.steps}
    print(f"[run_real] {skill.path} on the real robot ({robot.name}, speed {speed.name}) with {dict(a.arg)}; out {out}")
    try:
        result = run_episode(host, None, 0, skill.dir, out, ledger, vision, access, interface=skill.interface, limits=limits,
                             record=False, tag="real", args=dict(a.arg) or None)  # fmt: skip
    finally:
        host.close()
    print(json.dumps({k: result[k] for k in ("end", "returned", "steps", "robot_time_s", "think_s", "model_usd", "n_llm", "n_vision", "n_observe")}, indent=1))
    print(f"[run_real] files in {out}")


if __name__ == "__main__":
    main()
