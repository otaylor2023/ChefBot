"""Live base pose from mocap, through the onboard server. Read-only: nothing moves.

    ssh -L 8765:localhost:8765 tidybot01        # with the onboard server running there
    python -m chefbot.scripts.real_pose [--hz 5] [--seconds 30] [--url ws://127.0.0.1:8765]

It sends only heartbeats, and the server halts nothing on disconnect for a client that never
commanded (onboard/server.py), so the drivers' goals stay as they were.

Each line: the base's mocap pose (x, y in m, yaw in deg), how long since the mocap pose last changed,
the publisher's tracking flag, and the odometry pose in the mocap world if the base driver is up.
Every sample also goes to runs/real_pose/<time>.jsonl. At the end, a summary: how much of the time
the pose was live, its spread while live (jitter, if the robot stood still), and the longest
stretch without a live pose.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.environments.real.env import CONFIG, Localization, RealEnv, load_config

REPO = Path(__file__).resolve().parents[2]


def _pose(p) -> str:
    return "      -        -        -  " if p is None else f"{p.x:8.4f} {p.y:8.4f} {math.degrees(p.yaw):7.2f}"


def line(loc: Localization) -> str:
    age = "   -  " if loc.mocap_age_s is None else f"{loc.mocap_age_s * 1000:5.0f}ms"
    status = "LIVE" if loc.live else ("stale" if not loc.mocap_fresh else "lost ")
    return (f"{status:5}  mocap {_pose(loc.mocap_pose)}  age {age}  valid {int(loc.mocap_valid)}  "
            f"odom->world {_pose(loc.base_pose)}{'' if loc.localized else ' (not localized)'}")  # fmt: skip


def record(loc: Localization, t: float) -> dict:
    def pose(p):
        return None if p is None else [p.x, p.y, p.yaw]

    return {"t": t, "seq": loc.seq, "live": loc.live, "mocap_pose": pose(loc.mocap_pose), "mocap_valid": loc.mocap_valid,
            "mocap_fresh": loc.mocap_fresh, "mocap_age_s": loc.mocap_age_s, "base_pose": pose(loc.base_pose),
            "localized": loc.localized, "state_age_s": loc.state_age_s}  # fmt: skip


def summarize(samples: list[dict]) -> str:
    """Live fraction, spread of the live poses, and the longest run of samples without a live pose."""
    if not samples:
        return "no samples"
    live = [s for s in samples if s["live"]]
    out = [f"{len(samples)} samples, {len(live)} live ({100.0 * len(live) / len(samples):.0f}%)"]
    if live:
        poses = np.array([s["mocap_pose"] for s in live])
        yaw = np.unwrap(poses[:, 2])
        out.append(f"live pose spread (std): x {poses[:, 0].std() * 1000:.2f} mm, y {poses[:, 1].std() * 1000:.2f} mm, "
                   f"yaw {math.degrees(yaw.std()):.3f} deg")  # fmt: skip
        out.append(f"live pose mean: x {poses[:, 0].mean():.4f} m, y {poses[:, 1].mean():.4f} m, "
                   f"yaw {math.degrees(math.atan2(np.sin(yaw).mean(), np.cos(yaw).mean())):.2f} deg")  # fmt: skip
    gap = longest = 0.0
    for prev, cur in zip(samples, samples[1:]):
        gap = 0.0 if cur["live"] else gap + (cur["t"] - prev["t"])
        longest = max(longest, gap)
    out.append(f"longest stretch without a live pose: {longest:.2f} s")
    ages = [s["mocap_age_s"] for s in samples if s["mocap_age_s"] is not None]
    if ages:
        out.append(f"mocap age: median {np.median(ages) * 1000:.0f} ms, max {max(ages) * 1000:.0f} ms (state sampled at the server's rate)")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=None, help="the onboard server (default: envs/real/config.yaml's url)")
    parser.add_argument("--hz", type=float, default=5.0, help="lines printed per second")
    parser.add_argument("--sample-hz", type=float, default=50.0, help="samples logged per second")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: until Ctrl-C)")
    parser.add_argument("--log", default=None, help="the JSONL log (default: runs/real_pose/<time>.jsonl)")
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else REPO / "runs" / "real_pose" / f"{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Arm poses are never sent from here, so the arm controller's point does not matter.
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"])
    print(f"connected; logging to {log_path}", flush=True)
    samples: list[dict] = []
    t0 = time.monotonic()
    next_print = t0
    try:
        with log_path.open("w") as log:
            while not args.seconds or time.monotonic() - t0 < args.seconds:
                t = time.monotonic() - t0
                loc = env.localization()
                samples.append(record(loc, t))
                log.write(json.dumps(samples[-1]) + "\n")
                if time.monotonic() >= next_print:
                    print(f"{t:7.2f}s  {line(loc)}", flush=True)
                    next_print += 1.0 / args.hz
                time.sleep(1.0 / args.sample_hz)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
    print("\n" + summarize(samples))
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
