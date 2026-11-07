"""Measure the end effector's load (mass and centre of mass) with the arm's own joint torque sensors: no scale.

Held still, a joint's measured torque (tau_J) is the true gravity torque of everything beyond it; libfranka's model
(`gravity`) computes it with the load Desk is configured with. Their difference is the gravity torque of the load
Desk is missing: a point mass dm at p in the flange frame, plus each sensor's small offset. Across poses that point
the wrist different ways, a least-squares fit gives dm, p and the offsets; the true load is Desk's plus that.

    python -m chefbot.onboard.calibration.load record out.json                 # moves the arm: Enter once, then all
                                                                               # the poses (Ctrl-C holds the arm there)
    python -m chefbot.onboard.calibration.load record out.json --resume        # on from the poses out.json has
    python -m chefbot.onboard.calibration.load solve out.json --desk-mass 0.73 --desk-com -0.01 0 0.03 \
        --tcp-z 0.13065 --json endeffector-config_chefbot.json                 # the fit, and a Desk file with it

The base must be level (gravity along the arm base's -z).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration import fr3

G = 9.81
READY = [0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785]
# Poses around ready (joint index -> angle, rad; the rest as ready): the wrist pointing down, sideways, tilted, turned.
POSES = [
    {}, {6: 1.985}, {6: -0.415}, {5: 2.1}, {5: 0.9}, {4: 0.9}, {4: -0.9},
    {4: 0.8, 5: 2.0, 6: 1.1}, {3: -1.6, 5: 1.9}, {0: 0.5, 4: -0.6, 5: 1.0, 6: 2.4},
    # the arm reaching out and pulled in (so the load's torques at the shoulder and elbow change, apart from the offsets)
    # (11 and 14 turned 0.4 rad at joint 1, 2026-11-06: as first written their elbow sat 6-7 mm inside the pole camera's
    # keep-out, added 11-03; turned, it clears by 1.7 / 1.8 cm and the shoulder and elbow load as before)
    {0: 0.4, 1: -0.3, 3: -1.9}, {1: -1.1, 3: -2.6, 5: 1.6}, {1: 0.1, 3: -1.4, 5: 1.4},
    {0: 0.4, 1: -0.4, 3: -1.7, 4: 0.7, 5: 2.2, 6: 0.2},
]
# Every pose keeps the fingertips 0.43-0.89 m above the arm base, at most 0.57 m ahead of it and 0.3 m to its left
# (checked with fr3.py): clear the space in front of the robot to that height.
HAND_INERTIA = [0.001, 0, 0, 0, 0.0025, 0, 0, 0, 0.0017]  # the Franka Hand's (Desk's default file)
HAND_ROTATION = [0.7071, -0.7071, 0, 0, 0.7071, 0.7071, 0, 0, 0, 0, 1, 0]  # 45 deg about the flange's z


def regressor(q) -> np.ndarray:
    """7 x 11: the joint torques that hold a load (dm, dm p in the flange frame) and the sensor offsets."""
    T, J = fr3.flange(q), fr3.jacobian(q)
    up = np.array([0.0, 0.0, G])  # the force that holds a unit mass (base frame)
    skew = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    col_m = J[:3].T @ up
    col_p = -J[3:].T @ skew @ T[:3, :3]  # moment (R dm p) x up
    return np.hstack([col_m[:, None], col_p, np.eye(7)])


def record(out: Path, seconds: float, resume: bool = False) -> None:
    from chefbot.onboard.calibration.arm import Arm, Stop

    arm = Arm()
    samples = json.loads(out.read_text())["samples"] if resume and out.exists() else []
    if samples:
        print(f"{len(samples)} poses already in {out}: on from pose {len(samples) + 1}")
    first = True
    try:
        for i, change in enumerate(POSES):
            if i < len(samples):
                continue
            q = np.array(READY)
            for j, v in change.items():
                q[j] = v
            arm.move_to(q, f"pose {i + 1}/{len(POSES)}", confirm=first)  # Enter for the first move of the run only
            first = False
            s = arm.still(seconds)
            samples.append(s)
            print(f"  recorded ({s['n']} readings; torque sd up to {max(s['tau_sd']):.2f} N m)")
            out.write_text(json.dumps({"samples": samples}, indent=1))
    except (Stop, KeyboardInterrupt):
        arm.hold()
        print("stopped: holding")
    finally:
        out.write_text(json.dumps({"samples": samples}, indent=1))
        print(f"{len(samples)} poses in {out}")


def solve(path: Path, desk_mass: float, desk_com, tcp_z: float | None, json_out: Path | None) -> dict:
    samples = json.loads(path.read_text())["samples"]
    A = np.vstack([regressor(s["q"]) for s in samples])
    r = np.concatenate([np.array(s["tau_J"]) - np.array(s["gravity"]) for s in samples])
    x, *_ = np.linalg.lstsq(A, r, rcond=None)
    fit = A @ x
    dm, mp, bias = float(x[0]), x[1:4], x[4:]
    m = desk_mass + dm
    com = (desk_mass * np.asarray(desk_com, float) + mp) / m
    rms_before, rms_after = float(np.sqrt(np.mean(r**2))), float(np.sqrt(np.mean((r - fit) ** 2)))
    out = {"poses": len(samples), "load_mass_kg": round(m, 3), "load_com_flange_m": np.round(com, 4).tolist(), "desk_missing_kg": round(dm, 3),
           "sensor_offsets_Nm": np.round(bias, 3).tolist(), "residual_rms_Nm": {"desk_config": round(rms_before, 3), "fitted": round(rms_after, 3)},
           "condition": float(np.linalg.cond(A))}  # fmt: skip
    print(json.dumps(out, indent=1))
    if json_out is not None:
        T = list(HAND_ROTATION) + [0.0, 0.0, float(tcp_z if tcp_z is not None else 0.1034), 1.0]
        json_out.write_text(json.dumps({"mass": round(m, 3), "centerOfMass": np.round(com, 4).tolist(), "transformation": T, "inertia": HAND_INERTIA}))
        print(f"Desk end-effector file: {json_out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("record")
    a.add_argument("out", type=Path)
    a.add_argument("--seconds", type=float, default=2.0)
    a.add_argument("--resume", action="store_true", help="keep the poses already in out and record the rest")
    b = sub.add_parser("solve")
    b.add_argument("data", type=Path)
    b.add_argument("--desk-mass", type=float, required=True, help="the mass Desk is configured with now (kg)")
    b.add_argument("--desk-com", type=float, nargs=3, required=True, help="Desk's centre of mass now (flange frame, m)")
    b.add_argument("--tcp-z", type=float, default=None, help="the TCP along the flange's z for the Desk file (m)")
    b.add_argument("--json", type=Path, default=None, help="write a Desk end-effector file with the measured load")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.out, args.seconds, args.resume)
    else:
        solve(args.data, args.desk_mass, args.desk_com, args.tcp_z, args.json)


if __name__ == "__main__":
    main()
