"""Run a fridge-lab skill on the real TidyBot++ (or with --dry-run, on fake hardware).

    # bring-up: connect, print the robot's state, save one image per camera; nothing moves
    python -m experiments.fridge_lab.real.run_real check [--dry-run]

    # one episode of a skill, exactly as the lab runs it (sandboxed process, same robot API)
    python -m experiments.fridge_lab.real.run_real run SKILL_DIR [--models gemini|astra] [--speed 0.5] [--dry-run]

On the NUC first: tidybot2-arx's base_server.py, and real/nuc/arx_arm_server.py (in place of
arm_server.py). Results go to --out (default ~/fridge_lab/real_runs/<time>): result.json, video.mp4,
keyframes.jpg, skill_log.txt, images/, calls.jsonl, trace.jsonl, astra.jsonl. Ctrl-C stops the robot
where it is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import yaml

from experiments.fridge_lab.real.host import RealHost

CONFIG = Path(__file__).resolve().parent / "robot_real.yaml"


def load_config(path: Path, args) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["dry_run"] = bool(args.dry_run)
    if getattr(args, "speed", None) is not None:
        cfg["safety"]["speed"] = float(args.speed)
    return cfg


def check(args) -> None:
    cfg = load_config(args.config, args)
    out = Path(args.out or Path.home() / "fridge_lab" / "real_runs" / time.strftime("check_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    host = RealHost(cfg, out, record=False)
    try:
        host.odom = host._root_in_vehicle(host.base.pose())  # odom here; the control loop never starts, so nothing moves
        p = host.proprio()
        print(json.dumps({k: p[k] for k in ("base_pose_odom", "arm_q", "gripper_width_m")}, indent=1))
        print("flange in odom:\n", np.round(np.array(p["flange_pose_odom"]), 3))
        for name, c in host.rgbd().items():
            iio.imwrite(out / f"{name}.png", c["rgb"])
            d = c["depth"]
            print(f"{name}: {c['model']}, K fx={c['K'][0, 0]:.1f}, depth {'none' if d is None else f'{(d > 0).mean():.0%} valid'}, "
                  f"camera at {np.round(c['cam_to_odom'][:3, 3], 3).tolist()} in odom")  # fmt: skip
        if host.uncalibrated():
            print("NOT CALIBRATED:", ", ".join(host.uncalibrated()))
        print("images in", out)
    finally:
        host.close()


def run(args) -> None:
    from experiments.fridge_lab import episode as E
    from experiments.fridge_lab import models as M

    E.MODELS = args.models
    cfg = load_config(args.config, args)
    out = Path(args.out or Path.home() / "fridge_lab" / "real_runs" / time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    host = RealHost(cfg, out, record=not args.no_video)
    try:
        missing = host.uncalibrated()
        if missing and not (args.dry_run or args.uncalibrated_ok):
            sys.exit(f"not calibrated: {', '.join(missing)} (robot_real.yaml); --uncalibrated-ok to run anyway")
        print("Resetting: base odometry to zero, arm to its ready pose.")
        host.reset_robot()
        if not (args.dry_run or args.yes):
            input(f"Speed {cfg['safety']['speed']}. Clear the area, put the robot ~1 m in front of the fridge door, facing it.\n"
                  "Enter to start (Ctrl-C stops the robot at any time): ")  # fmt: skip
        limits = {"control_steps": int(float(cfg["safety"]["max_episode_s"]) * 15), "model_usd": args.model_usd}
        ledger = M.Ledger(out / "model_ledger.jsonl", args.model_usd)
        vision = M.Vision(args.vision_url or os.environ.get("FRIDGE_VISION_URL"))
        scene = {"id": args.label, "door": 0}
        try:
            result = E.run_skill_episode(host, scene, 0, Path(args.skill), out, ledger, vision, limits=limits, record=not args.no_video,
                                         tag=args.label)  # fmt: skip
        except KeyboardInterrupt:
            host.halt()
            print("\nStopped.")
            raise
        result["real"] = {"config": {k: cfg[k] for k in ("arm", "base", "safety")}, "uncalibrated": missing, "fault": repr(host.fault) if host.fault else None}
        (out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result.get(k) for k in ("end", "robot_time_s", "n_llm", "n_astra", "model_usd")}))
        print("results in", out)
    finally:
        host.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--out", type=Path)
        p.add_argument("--dry-run", action="store_true", help="fake arm, base and cameras: test the plumbing without a robot")
        if name == "run":
            p.add_argument("skill", type=Path)
            p.add_argument("--models", choices=("gemini", "astra"), default="gemini", help="what the skill may call (its lab's)")
            p.add_argument("--speed", type=float, help="override safety.speed")
            p.add_argument("--model-usd", type=float, default=0.10)
            p.add_argument("--vision-url", help="the vision server (OWLv2/DINO/SAM); default $FRIDGE_VISION_URL")
            p.add_argument("--label", default="real")
            p.add_argument("--no-video", action="store_true")
            p.add_argument("--uncalibrated-ok", action="store_true", help="run with placeholder calibration (wrist camera)")
            p.add_argument("--yes", action="store_true", help="don't wait for Enter before starting")
    args = ap.parse_args()
    {"check": check, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
