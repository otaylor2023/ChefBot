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
from experiments.fridge_lab.kitchen import ROBOT, LabHost

STANDOFF = 0.12
PULL_DEG_PER_STEP = 0.5
PULL_TO = 88.0
D_OPEN, D_CLOSED = ROBOT.pad_depth
# Where on a vertical bar to grasp, and the approach's upward pitch there: the Franka grasps at
# 1.05 m, level; the YAM reaches level only to about 0.95 m, so it grasps low and tilts up above that.
GRASP_Z = 1.05 if ROBOT.robocasa == "PandaOmron" else 0.92
LEVEL_MAX_Z, PITCH_UP = (9.0, 0.0) if ROBOT.robocasa == "PandaOmron" else (0.94, math.radians(40.0))


def _rot_about(axis: np.ndarray, point: np.ndarray, theta: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * theta).as_matrix()
    M[:3, 3] = point - M[:3, :3] @ point
    return M


def _row(host: LabHost, T: np.ndarray, g: float, base_world: np.ndarray) -> list[float]:
    """11-D row from a world flange pose, gripper, and a world robot-root pose."""
    F = np.linalg.inv(host.odom) @ T
    B = np.linalg.inv(host.odom) @ base_world
    return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), g, float(B[0, 3]), float(B[1, 3]), float(math.atan2(B[1, 0], B[0, 0]))]


def _pads_on(host: LabHost, D) -> bool:
    pads = set().union(*host.env._pad_geoms.values())
    hg = set(D.handle_geoms)
    return any((a in pads and b in hg) or (b in pads and a in hg) for a, b in host.env._touching())


def run(host: LabHost, door: int, seed: int, pull_to: float = PULL_TO, verbose: bool = False) -> dict:
    nominal = host.find_start(door, seed, jitter=False, outs=ROBOT.grasp_outs)  # where to stand: a clear stance, no jitter
    pose = host.find_start(door, seed)
    if pose is None or nominal is None:
        return {"door": door, "seed": seed, "door_deg": 0.0, "max_deg": 0.0, "steps": 0, "start": None}
    D = host.doors[door]
    vertical = (D.handle_hi[2] - D.handle_lo[2]) > max(D.handle_hi[:2] - D.handle_lo[:2])
    z = float(np.clip(GRASP_Z, D.handle_lo[2] + 0.04, D.handle_hi[2] - 0.04)) if vertical else float(D.handle[2])
    grip = np.array([D.handle[0], D.handle[1], z])
    pitch = PITCH_UP if z > LEVEL_MAX_Z else 0.0
    approach = math.cos(pitch) * -D.out + math.sin(pitch) * np.array([0.0, 0.0, 1.0])
    closing = D.along if vertical else np.array([0.0, 0.0, 1.0])
    # Step to the nominal stance first (arm held), so every seed grasps from the same place.
    base_nominal = host.arm.frames.arm_base_in_world(nominal)
    F_rel = host.flange_pose()  # keep the arm as it is relative to the base
    if ROBOT.robocasa == "PandaOmron":
        for _ in range(45):
            host.step(_row(host, base_nominal @ F_rel, 0.0, base_nominal))
    else:
        # The YAM is too short to hold the flange still while the base drives: carry it with the base, from
        # its start about 1 m out until the base has arrived.
        for i in range(150):
            r = host.step(_row(host, host.root() @ F_rel, 0.0, base_nominal))
            if i >= 30 and r["base_error_m"] < 0.01 and r["base_yaw_error_rad"] < 0.02:
                break
    root0 = host.root()
    # The two grasps 180 deg apart about the approach: take one the arm reaches (the YAM's wrist turns
    # only +-120 deg), the one nearer the wrist's current orientation if both.
    from scipy.spatial.transform import Rotation

    R_now = (host.root() @ host.flange_pose())[:3, :3]
    def _turn(c):
        return np.linalg.norm(Rotation.from_matrix(flange_for_pads(grip, approach, c, D_OPEN)[:3, :3] @ R_now.T).as_rotvec())

    def _margin(c):
        """Smallest distance (rad) of the grasp's IK solution from a joint limit; -1 if unreachable."""
        T = np.linalg.inv(root0) @ flange_for_pads(grip, approach, c, D_OPEN)
        q = host.arm.ik.solve_pose(host.tcp_from_flange(T), seed=host.env.get_state().arm_q)
        lim = host.arm.ik.joint_limits
        return -1.0 if q is None else float(np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q)))

    # Prefer the grasp that keeps the wrist clear of its limits (stepping toward one at a limit stalls).
    options = sorted((closing, -closing), key=_turn)
    margins = [_margin(c) for c in options]
    # IK from the current joints can miss a reachable grasp (margin -1), so keep the nearer turn unless only the other
    # one is clearly clear of the limits.
    closing = options[1] if margins[1] >= 0.3 and margins[0] < 0.3 else options[0]
    if verbose:
        print({"grip": grip.round(3).tolist(), "pitch_deg": round(math.degrees(pitch)), "closing": closing.round(2).tolist(),
               "margins": [round(m, 2) for m in margins]}, flush=True)  # fmt: skip

    def row(T, g, base_world):
        return _row(host, T, g, base_world)

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
