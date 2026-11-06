"""Parking for counter work targets by the pick task's start rule, and its trials on the real base: each target in
turn, the stance chosen (robot/nav/goal.py counter_stance), driven there (Robot(RealEnv).go_to_pose, through
scripts/real_nav.py), and where the base ended reported against the rule (park_report).

    python -m chefbot.scripts.real_park --map maps/src_kitchen --counter island --random 8 [--seed 0]
    python -m chefbot.scripts.real_park --map maps/src_kitchen --target X Y [--target X Y ...] [--plan-only]

Trials: --sides (one side of a counter, e.g. against a wall), --x-range (one stretch), --mix COUNTER:SIDE ...
(random A to B across the kitchen), --alternate COUNTER SIDE (back and forth between two counters), --end-margin,
--hold (hold on arrival), --clearance / --turn-in-room (the path style). A target whose path needs a turn the
base may not make on the way (a heading zone, a tight aisle) turns first where the base is. A link stall or a
brief mocap dropout retries the same target (the robot layer's transient_retries, twice). The session stops at the
first move that does not end `reached`, and on a rotation more than 15 deg beyond the short way. Every move is
logged under runs/real_park/<time>/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from chefbot.environments.real.env import CONFIG, RealEnv, load_config
from chefbot.interface.types import Pose2D
from chefbot.robot.nav.goal import counter_stance
from chefbot.scripts.real_nav import REPO, add_go_args, base_speeds, check_server_cap, embodiment, go_planned, map_robot, robot_stop, wrap


def counter_box(map_dir: Path, name: str | None = None) -> tuple:
    """(centre, size, yaw rad, name) of a box obstacle in the map's spec (the first one unless named)."""
    import yaml

    spec = yaml.safe_load((Path(map_dir) / "spec.yaml").read_text())
    for o in spec.get("obstacles") or []:
        if "box" in o and (name is None or o["name"] == name):
            b = o["box"]
            return tuple(b["center"]), tuple(b["size"]), math.radians(b.get("yaw_deg", 0.0)), o["name"]
    raise SystemExit(f"no box obstacle {name or ''} in {map_dir}/spec.yaml")


def random_targets(box, n: int, seed: int, sides=None, x_range=None, end_margin=None) -> list[tuple[float, float]]:
    """Targets drawn as the pick task draws them: a side, a spot along it >= 0.35 m from its ends, 0.10-0.16 m in.
    `sides`: only these (south / north / east / west), e.g. a counter against a wall reached from one side."""
    from chefbot.robot.nav.goal import PARK_END_MARGIN_M, TRAINED_INSET_M, counter_faces

    rng = np.random.default_rng(seed)
    faces = [f for f in counter_faces(*box[:3]) if not sides or f["name"] in sides]
    if not faces:
        raise SystemExit(f"no counter side among {sides}")
    out, tries = [], 0
    while len(out) < n:  # `x_range` (world x min, max): only targets in this stretch, e.g. one section of a kitchen
        tries += 1
        if tries > 1000 * max(n, 1):
            raise SystemExit(f"no targets on {sides or 'any side'} within x {x_range}")
        f = faces[int(rng.integers(len(faces)))]
        along = rng.uniform(-1, 1) * (f["half_along"] - (PARK_END_MARGIN_M if end_margin is None else end_margin))
        inset = rng.uniform(*TRAINED_INSET_M)
        p = np.asarray(box[0], dtype=float) + f["t"] * along + f["n"] * (f["half_n"] - inset)
        if x_range and not (x_range[0] <= p[0] <= x_range[1]):
            continue
        out.append((round(float(p[0]), 3), round(float(p[1]), 3)))
    return out


def turn_taken_deg(log_dir: Path) -> float:
    """The signed rotation the base actually made over a move's logs (unwrapped mocap yaw)."""
    yaws = [json.loads(line)["pose"][2] for f in sorted(Path(log_dir).glob("leg*.jsonl")) for line in f.open()]
    return float(np.degrees(np.unwrap(yaws)[-1] - yaws[0])) if yaws else 0.0


def park(env: RealEnv, args) -> list[dict]:
    """Each target in turn: choose the stance, drive there (one smooth path, or the stations' straight-in), report
    where the base ended against the start rule. Stops the session at the first move that does not end `reached`,
    and on a rotation more than 15 deg beyond the short way."""
    from chefbot.robot.nav.goal import park_report

    box = counter_box(args.map, args.counter)
    targets = [tuple(t) for t in args.target or []] + random_targets(box, args.random, args.seed, args.sides, args.x_range, args.end_margin)
    boxes = [box] * len(targets)
    if args.mix:  # random places across several counter sides: each target on a randomly chosen one
        rng = np.random.default_rng(args.seed + 7)
        pairs = [m.split(":") for m in args.mix]
        mixed, mixed_boxes = [], []
        for k in range(args.random):
            name, side = pairs[int(rng.integers(len(pairs)))]
            b = counter_box(args.map, name)
            mixed += random_targets(b, 1, args.seed * 100 + k, [side], args.x_range, args.end_margin)
            mixed_boxes.append(b)
        targets, boxes = mixed, mixed_boxes
    if args.alternate:  # back and forth between two counters: A, B, A, B ... (B's targets from its own side(s))
        box_b = counter_box(args.map, args.alternate[0])
        tb = random_targets(box_b, args.random, args.seed + 1000, args.alternate[1:], args.x_range, args.end_margin)
        ta = targets[len(targets) - args.random:]
        mixed, mixed_boxes = [tuple(t) for t in args.target or []], [box] * len(args.target or [])
        for a, b in zip(ta, tb):
            mixed += [a, b]
            mixed_boxes += [box, box_b]
        targets, boxes = mixed, mixed_boxes
    if not targets:
        raise SystemExit("no targets: --target X Y and/or --random N")
    from chefbot.scripts.real_map import load as load_map

    grid, _ = load_map(args.map)
    stop_model, accel = robot_stop()
    robot = map_robot(env, args, grid, settle_s=0.2, still_mps=0.01, arrive_hold=args.hold,
                      path_clearance=args.clearance, turn_in_room=args.turn_in_room)
    robot._cfg = replace(robot._cfg, transient_retries=2)  # a link stall or a brief mocap dropout: the same target again
    print(f"stop model: tau {stop_model.tau_s:.3f} s, brake {stop_model.brake_mps2:.2f} m/s^2 "
          f"({1000 * stop_model.distance(args.speed):.0f} mm from {args.speed:.3f} m/s); driver accel {accel} m/s^2")
    session = REPO / "runs" / "real_park" / f"{datetime.now():%Y%m%d_%H%M%S}"
    session.mkdir(parents=True, exist_ok=True)
    (session / "session.json").write_text(json.dumps({"targets": targets, "args": {k: (str(v) if isinstance(v, Path) else v)
                                                      for k, v in vars(args).items()}}, indent=2))  # fmt: skip
    results = []
    for i, target in enumerate(targets):
        box = boxes[i]
        pose, info = counter_stance(robot, box, target, args.out, args.approach, args.end_margin)
        head = f"[{i + 1}/{len(targets)}] target ({target[0]:.3f}, {target[1]:.3f})"
        if pose is None and info == "no stance reachable from here" and not args.plan_only:
            # A path that needs a turn where the base may not turn (a heading zone, a tight aisle): turn in place to
            # the stance's heading first, where the base is (if it may), then plan again.
            from chefbot.robot.nav.goal import counter_park_candidates

            cands = counter_park_candidates(*box[:3], target, out_m=tuple(args.out or (0.45, 0.48)))
            here = env.localization().mocap_pose
            if cands and here is not None:
                yaw = cands[0][0].yaw
                # Back away from what the base faces (into the aisle) and turn there; else turn where it is.
                for back in (0.35, 0.0):
                    turn = Pose2D(here.x - back * math.cos(here.yaw), here.y - back * math.sin(here.yaw), yaw)
                    print(f"{head}: no path with the turn on the way; {'backing ' + format(back, '.2f') + ' m and ' if back else ''}"
                          f"turning to {math.degrees(yaw):.0f} deg first")
                    tlog = session / f"{i:02d}_turn_first_{int(back * 100)}"
                    tlog.mkdir(parents=True, exist_ok=True)
                    rt = go_planned(env, turn, grid, robot=robot, log_dir=tlog, confirm=lambda _: None)
                    print(f"    turn: {rt['reason']}")
                    if rt.get("ok"):
                        pose, info = counter_stance(robot, box, target, args.out, args.approach, args.end_margin)
                        break
                    if rt.get("moved"):
                        break  # it moved and stopped: don't try another
        if pose is None:
            print(f"{head}: skipped, {info}")  # nothing moved: go on to the next target
            results.append({"target": target, "ok": False, "reason": info, "moved": False, "skipped": True})
            continue
        print(f"{head}: {info['side']} side, {info['inset_m']:.3f} m in, stance ({pose.x:.3f}, {pose.y:.3f}, "
              f"{math.degrees(pose.yaw):.1f} deg) {info['out_m']:.2f} m out" + ("" if info["trained"] else "  [inset outside 0.10-0.16]"))
        if args.plan_only:
            results.append({"target": target, "stance": [pose.x, pose.y, pose.yaw], **info, "ok": True, "reason": "planned"})
            continue
        start = env.localization().mocap_pose
        log_dir = session / f"{i:02d}_{info['side']}"
        log_dir.mkdir(parents=True, exist_ok=True)  # go_planned may refuse before it makes it
        t0 = time.monotonic()
        r = go_planned(env, pose, grid, robot=robot, log_dir=log_dir, confirm=lambda _: None, approach_m=args.approach)
        elapsed = time.monotonic() - t0
        end = env.localization().mocap_pose
        asked = math.degrees(wrap(pose.yaw - start.yaw))
        took = turn_taken_deg(log_dir)
        row = {"target": target, "stance": [pose.x, pose.y, pose.yaw], **info, "ok": bool(r.get("ok")), "reason": r["reason"],
               "seconds": elapsed, "path_m": r.get("path_m"), "final_error_m": r.get("final_error_m"),
               "final_yaw_error_deg": r.get("final_yaw_error_deg"), "turn_asked_deg": asked, "turn_taken_deg": took,
               "stops": [leg["reason"] for leg in r.get("legs", [])]}  # fmt: skip
        if end is not None:
            row["rule"] = park_report(*box[:3], info["side"], end, target)
        (log_dir / "park.json").write_text(json.dumps(row | {"result": r}, indent=2, default=str))
        results.append(row)
        rule = row.get("rule", {})
        nan = float("nan")
        err_m, err_deg = r.get("final_error_m"), r.get("final_yaw_error_deg")
        print(f"    {r['reason']} in {elapsed:.1f} s, {1000 * (nan if err_m is None else err_m):.1f} mm, "
              f"{nan if err_deg is None else err_deg:+.2f} deg; turned {took:+.0f} (asked {asked:+.0f}); "
              f"out {rule.get('out_m', float('nan')):.3f} m, target {rule.get('target_lateral_m', float('nan')) * 1000:+.0f} mm "
              f"off centre, {rule.get('yaw_off_square_deg', float('nan')):+.2f} deg off square")  # fmt: skip
        if abs(took) > abs(asked) + 15.0:
            print(f"    STOP: the base turned {took:+.0f} deg for {asked:+.0f}: not the short way")
            break
        if not r.get("ok"):
            break
    (session / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"session: {session}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None)
    k = ap
    k.add_argument("--target", type=float, nargs=2, action="append", metavar=("X", "Y"), help="a target on the counter (world)")
    k.add_argument("--random", type=int, default=0, help="this many targets drawn by the start rule, one after another")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--sides", nargs="+", choices=("south", "north", "east", "west"), default=None,
                   help="draw --random targets on these counter sides only (a counter against a wall)")
    k.add_argument("--clearance", type=float, default=0.0,
                   help="m: plan paths that keep this far from obstacles where they can (down the middle of aisles)")
    k.add_argument("--turn-in-room", action="store_true", help="turn where the path has room, not evenly along it")
    k.add_argument("--mix", nargs="+", default=None, metavar="COUNTER:SIDE",
                   help="random targets, each on a randomly chosen COUNTER:SIDE of these (random A to B across the kitchen)")
    k.add_argument("--hold", action="store_true",
                   help="once inside --tol / --yaw-tol, hold the base where it is (no re-aiming for millimetres)")
    k.add_argument("--alternate", nargs="+", default=None, metavar=("COUNTER", "SIDE"),
                   help="alternate --random targets with as many on this second counter (its name, then its side(s))")
    k.add_argument("--end-margin", type=float, default=None,
                   help="m: targets (and stances) this close to a counter side's ends (default: the pick rule's 0.35)")
    k.add_argument("--x-range", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                   help="draw --random targets only within this world-x stretch (one section of a kitchen)")
    k.add_argument("--counter", default=None, help="the counter's obstacle name in the map spec (default: the first box)")
    k.add_argument("--plan-only", action="store_true", help="choose and print the stances; nothing moves")
    k.add_argument("--out", type=float, nargs="+", default=None,
                   help="m, base centre out from the edge, first preferred (default 0.45 0.48: the stations' closeness)")
    k.add_argument("--approach", type=float, default=0.0,
                   help="m straight in at the end (0: one smooth path straight to the stance; 0.25: the stations' way)")
    add_go_args(k)
    args = ap.parse_args()
    if args.map is None:
        raise SystemExit("park needs --map")
    args.speed, args.yaw_speed, args.profile = base_speeds(embodiment(), args.profile, args.speed, args.yaw_speed)
    print(f"speed profile {args.profile}: {args.speed:.3f} m/s, {args.yaw_speed:.3f} rad/s")
    env = RealEnv.from_config(np.eye(4), url=args.url or load_config(CONFIG)["url"], require_arm=False)  # arm poses are never sent
    try:
        try:
            check_server_cap(env.server_limits, args.speed, args.yaw_speed)
        except ValueError as exc:
            raise SystemExit(f"refusing: {exc}") from None
        t0 = time.monotonic()  # the server needs two mocap frames to call the pose live
        while not env.localization().live and time.monotonic() - t0 < 3.0:
            time.sleep(0.05)
        if not args.plan_only:
            input(f"Park for {len(args.target or []) + args.random} target(s), one after another, by A* on {args.map}. "
                  "Floor clear and e-stop in hand? Enter to drive, Ctrl-C to cancel: ")  # fmt: skip
        park(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
