"""A scripted fridge opener on the same whole-body controller: does the door open
by contact at all? (The fridge's R-GT: it uses ground-truth door geometry.)

    python -m experiments.astranomy.fridge_oracle --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.astranomy.fridge import FridgeHost
from experiments.astranomy.host import REPO, quat_wxyz

GRASP_Z = 1.10  # world height on the handle bar to hold
STANDOFF = 0.12  # m in front of the bar before closing
PULL_DEG_PER_STEP = 0.5
PULL_TO = 75.0


def odom_pose(host: FridgeHost, T_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(host.odom) @ T_world


def flange_for_pads(pads_world: np.ndarray, approach: np.ndarray, closing: np.ndarray, d: float) -> np.ndarray:
    """World flange pose (DROID axes: +X along the fingers, Y across the pads) putting the pad midpoint at `pads_world`."""
    X = approach / np.linalg.norm(approach)
    Y = closing - (closing @ X) * X
    Y /= np.linalg.norm(Y)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([X, Y, np.cross(X, Y)])
    T[:3, 3] = pads_world - d * X
    return T


def row(host: FridgeHost, T_flange_world: np.ndarray, gripper: float, base_world: np.ndarray) -> list[float]:
    F = odom_pose(host, T_flange_world)
    B = odom_pose(host, base_world)
    return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), gripper, float(B[0, 3]), float(B[1, 3]), float(math.atan2(B[1, 0], B[0, 0]))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = REPO / "experiments" / "astranomy" / "results" / "fridge_oracle" / f"seed{args.seed}"
    host = FridgeHost(out)
    host.reset_fridge(args.seed)
    m, d = host.env.model, host.env.data
    hinge = d.xanchor[host.door_joint].copy()
    handle0 = d.geom_xpos[host.handle_geom].copy()
    grip0 = np.array([handle0[0], handle0[1], GRASP_Z])
    approach0 = np.array([0.0, 1.0, 0.0])  # into the door (world +y)
    closing0 = np.array([1.0, 0.0, 0.0])  # across the bar's width
    d_open, d_closed = 0.1367, 0.1477
    root0 = host.root()
    log = []

    def go(T_flange, grip, base_world, steps, label):
        for _ in range(steps):
            r = host.step(row(host, T_flange, grip, base_world))
            host.observe()
        held = any(host.env.model.body_rootid[host.env.model.geom_bodyid[g]] == host.env.model.body_rootid[host.env.model.jnt_bodyid[host.door_joint]]
                   for g in [])  # fmt: skip
        log.append({"phase": label, "step": host.step_count, "door_deg": round(host.door_deg(), 1), "pos_err": round(r["pos_error_m"], 3),
                    "base_err": round(r["base_error_m"], 3), "base_contact": host.env.get_state().base_in_contact})  # fmt: skip
        print(log[-1], flush=True)

    # A: in front of the bar, open. B: in, close.
    pre = flange_for_pads(grip0 - STANDOFF * approach0, approach0, closing0, d_open)
    go(pre, 0.0, root0, 60, "pre-grasp")
    at = flange_for_pads(grip0, approach0, closing0, d_open)
    go(at, 0.0, root0, 30, "at handle")
    go(flange_for_pads(grip0, approach0, closing0, d_closed), 1.0, root0, 15, "closed")
    # C: pull: gripper and base turn together about the hinge (the door's own motion).
    T_grip = flange_for_pads(grip0, approach0, closing0, d_closed)
    theta = 0.0
    while theta < PULL_TO:
        theta += PULL_DEG_PER_STEP
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        about = np.eye(4)
        about[:3, 3] = [hinge[0], hinge[1], 0.0]
        M = about @ Rz @ np.linalg.inv(about)
        host.step(row(host, M @ T_grip, 1.0, M @ root0))
        host.observe()
        if host.step_count % 30 == 0:
            r = host.actions[-1]
            print({"theta_cmd": round(theta, 1), "door_deg": round(host.door_deg(), 1), "pos_err": round(r["pos_error_m"], 3),
                   "base_err": round(r["base_error_m"], 3), "base_contact": host.env.get_state().base_in_contact}, flush=True)  # fmt: skip
    for _ in range(15):
        host.step(row(host, M @ T_grip, 1.0, M @ root0))
    result = {"seed": args.seed, "door_deg": host.door_deg(), "success": host.success(), "steps": host.step_count}
    print(result)
    (out / "result.json").write_text(json.dumps(result))
    host.save_video()
    host.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
