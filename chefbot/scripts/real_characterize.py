"""Characterize the real base inside a fenced square: a planned set of moves, every one logged, then a report.

    python -m chefbot.scripts.real_characterize plan --half 0.2              # the moves, checked against the fence
    python -m chefbot.scripts.real_characterize run --half 0.2 [--blocks repeat,directions,turns,combo,speed]
    python -m chefbot.scripts.real_characterize report runs/characterize/<session>

Needs chefbot/onboard/bringup/base.sh on the robot and the port forwarded (as scripts/real_nav.py).

The fence is a square centred where the base stands when `run` starts (C), with sides along its heading.
`--half` is how far the base CENTRE may go from C, in each direction (m): for a 3 ft (0.91 m) square of clear
floor that the whole base must stay inside, that is about 0.91 / 2 - the base's half-width. Every target is
checked against the fence before anything moves; leaving it (plus 5 cm) stops the move. One Enter starts the
session; Ctrl-C stops it at any time; any move that does not end `reached` ends the session (the base is held).

Every move goes out from C and comes back to C. Blocks:
  repeat      the same move forward, 5 times                       -> repeatability (spread of where it ends)
  directions  8 directions at --step                               -> direction-dependent bias, wheel steering
  turns       in place: +-15, +-45, +-90, 180 deg                  -> yaw accuracy; centre drift (marker_xy check)
  combo       a step and a turn together                           -> coupling of translation and rotation
  speed       forward at 0.05, 0.08, 0.12 m/s                       -> overshoot and settling against speed
The report gives, per block, final errors, overshoot, time to 5 cm, peak speed, off-line, how far the base
rolled after the hold, the wheel odometry's distance over the true one, and calibration hints.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.envs.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D
from chefbot.scripts.real_nav import Fence, go, relative_point, wrap

REPO = Path(__file__).resolve().parents[2]
BLOCKS = ("repeat", "directions", "turns", "combo", "speed")
DIRECTIONS = (("forward", 1, 0), ("forward-left", 1, 1), ("left", 0, 1), ("back-left", -1, 1), ("back", -1, 0),
              ("back-right", -1, -1), ("right", 0, -1), ("forward-right", 1, -1))  # fmt: skip


def plan(half: float, step: float | None = None, blocks=BLOCKS, speed: float = 0.1) -> list[dict]:
    """The moves, each as (forward, left, turn) from C, and the speed to do it at."""
    step = step if step is not None else round(min(0.8 * half, 0.25), 3)
    moves = []
    if "repeat" in blocks:
        moves += [{"block": "repeat", "name": f"forward-{i + 1}", "forward": step, "left": 0.0, "turn": 0.0} for i in range(5)]
    if "directions" in blocks:
        for name, f, l in DIRECTIONS:
            n = math.hypot(f, l)
            moves.append({"block": "directions", "name": name, "forward": step * f / n, "left": step * l / n, "turn": 0.0})
    if "turns" in blocks:
        moves += [{"block": "turns", "name": f"turn{a:+d}", "forward": 0.0, "left": 0.0, "turn": float(a)} for a in (15, -15, 45, -45, 90, -90, 180)]
    if "combo" in blocks:
        moves += [{"block": "combo", "name": "forward+turn45", "forward": step, "left": 0.0, "turn": 45.0},
                  {"block": "combo", "name": "left+turn-45", "forward": 0.0, "left": step, "turn": -45.0},
                  {"block": "combo", "name": "back-right+turn90", "forward": -step / 2**0.5, "left": -step / 2**0.5, "turn": 90.0}]
    if "speed" in blocks:
        moves += [{"block": "speed", "name": f"forward@{v:.2f}", "forward": step, "left": 0.0, "turn": 0.0, "speed": v} for v in (0.05, 0.08, 0.12)]
    for m in moves:
        m.setdefault("speed", speed)
        if max(abs(m["forward"]), abs(m["left"])) > half + 1e-9:
            raise ValueError(f"{m['name']} leaves the fence (half {half} m)")
    return moves


def _pose(p: dict) -> Pose2D:
    return Pose2D(p["x"], p["y"], math.radians(p["yaw_deg"]))


def run(env: RealEnv, half: float, out: Path, moves: list[dict], yaw_speed: float = 0.3, confirm=input) -> list[dict]:
    loc = env.localization()
    if not loc.live:
        raise SystemExit("no live mocap pose")
    c = loc.mocap_pose
    centre = {"x": c.x, "y": c.y, "yaw_deg": math.degrees(c.yaw)}
    fence = Fence(c.x, c.y, math.degrees(c.yaw), half)
    out.mkdir(parents=True, exist_ok=True)
    (out / "moves").mkdir(exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"centre": centre, "half": half, "moves": moves, "started": datetime.now().isoformat()}, indent=2))
    print(f"C: x {c.x:.3f}  y {c.y:.3f}  yaw {math.degrees(c.yaw):.1f}; fence: centre within +-{half} m of C")
    print(f"{len(moves)} moves, each out from C and back; about {sum(2 * (5 + math.hypot(m['forward'], m['left']) / m['speed']) for m in moves) / 60:.0f} min")
    try:
        confirm("Floor clear inside the square and e-stop in hand? Enter to start the session, Ctrl-C to stop it: ")
    except (KeyboardInterrupt, EOFError):
        print("cancelled")
        return []
    results = []
    with (out / "results.jsonl").open("w") as rf:
        for i, m in enumerate(moves):
            for leg, target in (("out", relative_point(centre, m["forward"], m["left"], m["turn"])), ("back", centre)):
                tag = f"{i:02d}_{m['name']}_{leg}"
                print(f"[{i + 1}/{len(moves)}] {m['block']:10} {m['name']:18} {leg:4}", end="  ", flush=True)
                try:
                    r = go(env, _pose(target), speed=m["speed"], yaw_speed=yaw_speed, confirm=lambda _: None,
                           log_path=out / "moves" / f"{tag}.jsonl", fence=fence, quiet=True)  # fmt: skip
                except KeyboardInterrupt:
                    r = {"ok": False, "reason": "stopped (Ctrl-C)"}
                r.update(block=m["block"], name=m["name"], leg=leg, index=i, move=m)
                results.append(r)
                rf.write(json.dumps(r) + "\n")
                rf.flush()
                err = r.get("final_error_m")
                print(f"{r['reason']:10}" + (f"  {err * 1000:5.1f} mm  {r.get('final_yaw_error_deg', 0):+5.2f} deg" if err is not None else ""))
                if not r.get("ok"):
                    print(f"session stopped: {r['reason']}")
                    return results
    print(f"done: {out}")
    return results


# -- report -------------------------------------------------------------------------


def _local_error(r: dict, centre: dict) -> np.ndarray:
    """Where it ended minus the goal, in C's frame (forward, left), m."""
    yaw = math.radians(centre["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = r["end"][0] - r["goal"][0], r["end"][1] - r["goal"][1]
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def report(session: Path) -> str:
    plan_ = json.loads((session / "plan.json").read_text())
    centre = plan_["centre"]
    results = [json.loads(line) for line in (session / "results.jsonl").read_text().splitlines() if line.strip()]
    for r in results:  # where each move ended, from its log
        rows = [json.loads(line) for line in (session / "moves" / f"{r['index']:02d}_{r['name']}_{r['leg']}.jsonl").read_text().splitlines()]
        drive = [x for x in rows if x["phase"] == "drive"]
        r["end"] = (drive or rows)[-1]["pose"] if (drive or rows) else None
    lines = [f"session {session.name}: {len(results)} legs, {sum(r.get('ok', False) for r in results)} reached; "
             f"C x {centre['x']:.3f} y {centre['y']:.3f} yaw {centre['yaw_deg']:.1f}; fence +-{plan_['half']} m"]
    ok = [r for r in results if r.get("ok") and r.get("end")]

    def stat(vals, scale=1.0, fmt="{:.1f}"):
        vals = [v * scale for v in vals if v is not None]
        return "-" if not vals else f"mean {fmt.format(np.mean(vals))}, max {fmt.format(np.max(vals))}"

    for block in BLOCKS:
        rs = [r for r in ok if r["block"] == block]
        if not rs:
            continue
        lines.append(f"\n[{block}] {len(rs)} legs")
        lines.append(f"  final error      {stat([r['final_error_m'] for r in rs], 1000)} mm;  yaw {stat([abs(r['final_yaw_error_deg']) for r in rs], 1, '{:.2f}')} deg")
        lines.append(f"  overshoot        {stat([r.get('overshoot_m') for r in rs], 1000)} mm;  yaw {stat([r.get('yaw_overshoot_deg') for r in rs], 1, '{:.2f}')} deg")
        lines.append(f"  time to 5 cm     {stat([r.get('time_to_5cm_s') for r in rs], 1, '{:.2f}')} s;  total {stat([r.get('seconds') for r in rs], 1, '{:.1f}')} s")
        lines.append(f"  max off line     {stat([r.get('max_off_line_m') for r in rs], 1000)} mm;  rolled after hold {stat([r.get('moved_after_hold_m') for r in rs], 1000)} mm")
        lines.append(f"  peak speed       {stat([r.get('peak_speed_mps') for r in rs], 1, '{:.3f}')} m/s;  yaw rate {stat([r.get('peak_yaw_rate_dps') for r in rs], 1, '{:.1f}')} deg/s")
        if block == "repeat":
            outs = np.array([_local_error(r, centre) for r in rs if r["leg"] == "out"])
            if len(outs) >= 2:
                lines.append(f"  repeatability    where 'out' ended: spread (std) forward {outs[:, 0].std() * 1000:.1f} mm, "
                             f"left {outs[:, 1].std() * 1000:.1f} mm; mean error ({outs[:, 0].mean() * 1000:+.1f}, {outs[:, 1].mean() * 1000:+.1f}) mm")  # fmt: skip
        if block == "directions":
            for r in [r for r in rs if r["leg"] == "out"]:
                e = _local_error(r, centre)
                lines.append(f"    {r['name']:14} ended ({e[0] * 1000:+5.1f}, {e[1] * 1000:+5.1f}) mm from goal (forward, left)")
        if block == "turns":
            drift = [r.get("max_off_line_m") for r in rs]
            yaw_err = [(r["move"]["turn"], r["final_yaw_error_deg"]) for r in rs if r["leg"] == "out"]
            lines.append(f"  centre drift in turns: {stat(drift, 1000)} mm  (a marker_xy error shows up here)")
            for a, e in yaw_err:
                lines.append(f"    turn {a:+6.0f} deg: yaw error {e:+.2f} deg")
        if block == "speed":
            for r in rs:
                if r["leg"] == "out":
                    lines.append(f"    {r['move']['speed']:.2f} m/s: overshoot {1000 * (r.get('overshoot_m') or 0):.1f} mm, "
                                 f"peak {r.get('peak_speed_mps', 0):.3f} m/s, to 5 cm {r.get('time_to_5cm_s') or float('nan'):.2f} s")  # fmt: skip
    scale = [r["odom_over_mocap"] for r in ok if r.get("odom_over_mocap")]
    if scale:
        lines.append(f"\nwheel odometry distance / true distance: mean {np.mean(scale):.3f} (std {np.std(scale):.3f}, n {len(scale)})")
    hints = []
    turns = [r for r in ok if r["block"] == "turns"]
    if turns and max(r.get("max_off_line_m", 0) for r in turns) > 0.02:
        hints.append("the base centre moves > 2 cm in pure turns: re-check marker_xy (calibration/base.py spin)")
    outs = [r for r in ok if r["block"] == "directions" and r["leg"] == "out"]
    if outs:
        mean = np.mean([_local_error(r, centre) for r in outs], axis=0)
        if np.linalg.norm(mean) > 0.01:
            hints.append(f"all directions end offset the same way ({mean[0] * 1000:+.0f}, {mean[1] * 1000:+.0f}) mm: a frame offset, not noise")
    if scale and abs(np.mean(scale) - 1) > 0.03:
        hints.append(f"wheel odometry is {100 * (np.mean(scale) - 1):+.1f}% off in distance (slip or wheel radius); mocap corrects it")
    lines.append("\ncalibration hints: " + ("; ".join(hints) if hints else "none: errors look like noise"))
    failed = [r for r in results if not r.get("ok")]
    if failed:
        lines.append(f"stopped early: {failed[0]['name']} {failed[0]['leg']}: {failed[0]['reason']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--half", type=float, required=True, help="m the base centre may go from C, each way")
        p.add_argument("--step", type=float, default=None, help="m per move (default min(0.8 half, 0.25))")
        p.add_argument("--blocks", default=",".join(BLOCKS))
        p.add_argument("--speed", type=float, default=0.1)
        p.add_argument("--yaw-speed", type=float, default=0.3)
    r = sub.add_parser("report")
    r.add_argument("session", type=Path)
    args = ap.parse_args()
    if args.cmd == "report":
        text = report(args.session)
        (args.session / "report.txt").write_text(text)
        print(text)
        return
    if not (0 < args.half <= 0.5):
        raise SystemExit("--half must be in (0, 0.5] m")
    if not (0 < args.speed <= 0.12 and 0 < args.yaw_speed <= 0.4):
        raise SystemExit("--speed <= 0.12 m/s and --yaw-speed <= 0.4 rad/s (the driver's limits)")
    moves = plan(args.half, args.step, tuple(b.strip() for b in args.blocks.split(",")), args.speed)
    if args.cmd == "plan":
        for m in moves:
            print(f"{m['block']:10} {m['name']:18} forward {m['forward']:+.3f}  left {m['left']:+.3f}  turn {m['turn']:+6.1f}  at {m['speed']} m/s")
        print(f"{len(moves)} moves (each out from C and back), all within +-{args.half} m")
        return
    out = REPO / "runs" / "characterize" / datetime.now().strftime("%Y%m%d_%H%M%S")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"])
    try:
        run(env, args.half, out, moves, args.yaw_speed)
    finally:
        env.close()
    if (out / "results.jsonl").exists():
        text = report(out)
        (out / "report.txt").write_text(text)
        print("\n" + text)


if __name__ == "__main__":
    main()
