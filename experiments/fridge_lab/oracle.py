"""The privileged fridge opener (ground-truth hinge and handle) on any kitchen: the
feasibility filter for scenes, and the ceiling the skills are compared with.

    python -m experiments.fridge_lab.oracle --layout 11 --style 11 --door 0 --seed 0

Pre-grasp in front of the handle, close on it, then turn gripper and base rigidly
about the hinge (the door's own motion) to PULL_TO.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.astranomy.fridge_oracle import flange_for_pads
from experiments.astranomy.host import REPO, quat_wxyz
from experiments.fridge_lab.kitchen import LabHost

STANDOFF = 0.12
PULL_DEG_PER_STEP = 0.5
PULL_TO = 88.0
D_OPEN, D_CLOSED = 0.1367, 0.1477


def _rot_about(axis: np.ndarray, point: np.ndarray, theta: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * theta).as_matrix()
    M[:3, 3] = point - M[:3, :3] @ point
    return M


def _pads_on(host: LabHost, D) -> bool:
    pads = set().union(*host.env._pad_geoms.values())
    hg = set(D.handle_geoms)
    return any((a in pads and b in hg) or (b in pads and a in hg) for a, b in host.env._touching())


def run(host: LabHost, door: int, seed: int, pull_to: float = PULL_TO, verbose: bool = False) -> dict:
    pose = host.find_start(door, seed)
    if pose is None:
        return {"door": door, "seed": seed, "door_deg": 0.0, "max_deg": 0.0, "steps": 0, "start": None}
    D = host.doors[door]
    vertical = (D.handle_hi[2] - D.handle_lo[2]) > max(D.handle_hi[:2] - D.handle_lo[:2])
    z = float(np.clip(1.05, D.handle_lo[2] + 0.04, D.handle_hi[2] - 0.04)) if vertical else float(D.handle[2])
    grip = np.array([D.handle[0], D.handle[1], z])
    approach = -D.out
    closing = D.along if vertical else np.array([0.0, 0.0, 1.0])
    root0 = host.root()
    # The two grasps 180 deg apart about the approach: take the one nearer the wrist's current orientation.
    from scipy.spatial.transform import Rotation

    R_now = (host.root() @ host.flange_pose())[:3, :3]
    def _turn(c):
        return np.linalg.norm(Rotation.from_matrix(flange_for_pads(grip, approach, c, D_OPEN)[:3, :3] @ R_now.T).as_rotvec())
    if _turn(-closing) < _turn(closing):
        closing = -closing

    def row(T, g, base_world):
        F = np.linalg.inv(host.odom) @ T
        B = np.linalg.inv(host.odom) @ base_world
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), g, float(B[0, 3]), float(B[1, 3]), float(math.atan2(B[1, 0], B[0, 0]))]

    for T, g, n in [(flange_for_pads(grip - STANDOFF * approach, approach, closing, D_OPEN), 0.0, 60),
                    (flange_for_pads(grip, approach, closing, D_OPEN), 0.0, 30),
                    (flange_for_pads(grip, approach, closing, D_CLOSED), 1.0, 15)]:  # fmt: skip
        for _ in range(n):
            r = host.step(row(T, g, root0))
        if verbose:
            print({"phase_steps": n, "grip": g, "pos_err": round(r["pos_error_m"], 3), "rot_err": round(r["rot_error_rad"], 3), "door": round(host.door_deg(), 1),
                   "touching": host.debug()["robot_touching"], "pads_on_handle": _pads_on(host, D)}, flush=True)  # fmt: skip
    T_grip = flange_for_pads(grip, approach, closing, D_CLOSED)
    theta, best = 0.0, 0.0
    while theta < pull_to:
        theta += PULL_DEG_PER_STEP
        M = _rot_about(D.axis, D.hinge, math.radians(theta))
        host.step(row(M @ T_grip, 1.0, M @ root0))
        best = max(best, host.door_deg())
        if verbose and host.step_count % 30 == 0:
            r = host.actions[-1]
            print({"theta": theta, "door": round(host.door_deg(), 1), "pos_err": round(r["pos_error_m"], 3), "base_err": round(r["base_error_m"], 3),
                   "pads_on_handle": _pads_on(host, D), "touching": host.debug()["robot_touching"]}, flush=True)  # fmt: skip
    for _ in range(15):
        host.step(row(M @ T_grip, 1.0, M @ root0))
    return {"door": door, "seed": seed, "door_deg": host.door_deg(), "max_deg": max(best, host.door_deg()), "steps": host.step_count,
            "start": [pose.x, pose.y, pose.yaw]}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", type=int, required=True)
    ap.add_argument("--style", type=int, required=True)
    ap.add_argument("--door", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()
    out = REPO / "experiments" / "fridge_lab" / "results" / "oracle" / f"L{args.layout}_S{args.style}_d{args.door}_s{args.seed}"
    host = LabHost(args.layout, args.style, out, record=args.video)
    t0 = time.time()
    r = run(host, args.door, args.seed, verbose=True)
    r.update({"layout": args.layout, "style": args.style, "model": host.model_xml, "wall_s": time.time() - t0})
    print(json.dumps(r))
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(r, indent=1))
    if args.video:
        host.save_video()
    host.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
