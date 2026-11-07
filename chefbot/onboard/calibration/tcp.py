"""The fingertips' distance from the flange, by touch: the hand comes straight down onto a card of known pose until its
fingertips meet it (the joint torques say so), and the flange's height then, less the card's, is the fingertips' length.

    python -m chefbot.onboard.calibration.tcp touch [--touches 3] [--on counter|card]

--on counter (the default): the fingertips touch the counter 12 cm in front of the card (toward the robot), its height
there from the wrist camera's depth looking straight down from 0.25 m (good to ~0.3 mm that close: depth_check), a
plane fitted around the spot; --on card: the card itself, its plane from its corners (but its cardboard gives a little
under the fingertips: 2026-11-06, three touches 5.6 mm apart).

The card: the base card (ids 1, 2) lying flat on the counter, in the wrist camera's view from where the arm is (the
joint calibration's setup). Its plane comes from its corners in the colour image (environments/real/config.yaml's
wrist camera, no depth): sub-millimetre at 0.3 m. Each touch: the hand straight down over the card (within a few cm of
its centre, the fingers as they are: both fingertips land on it), 3 cm above where the fingertips should meet it
(robot.yaml body hand_m), then 1 mm steps down, the joints' external torque (measured less libfranka's gravity model)
watched against where it started; contact past CONTACT_NM ends the descent, and the hand goes back up. It never goes
more than MAX_BELOW below the expected contact. Enter before the first move. Run it with arm.sh (OpenSai holding the
arm, the server not running: the camera is needed).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

STEP = 0.001  # m per step down
ABOVE = 0.03  # m above the expected contact, to start
MAX_BELOW = 0.02  # m below it at most
CONTACT_NM = 1.2  # N m: the largest change in any of joints 2, 4, 6's external torque that counts as contact, on two
SETTLE_S = 0.4  # s after each step before the torques are read                          steps in a row
LOOK_FROM = 0.25  # m: the wrist camera above the counter's spot, looking straight down, for its height
IN_FRONT = 0.12  # m from the card's centre toward the robot: the counter's spot


def _ext(arm) -> np.ndarray:
    """The joints' external torque now: measured less libfranka's gravity model (averaged a few samples)."""
    from chefbot.onboard.calibration.arm import GRAVITY, TAU_J, _array

    out = []
    for _ in range(12):
        tau, g = _array(arm.r.get(TAU_J)), _array(arm.r.get(GRAVITY))
        if tau is not None and g is not None:
            out.append(tau - g)
        time.sleep(0.01)
    return np.mean(out, axis=0)


def _goto(arm, q, settle_s=SETTLE_S) -> None:
    """A small joint move (already checked): the goal, then wait it out."""
    from chefbot.onboard import keys as K

    arm.r.set(K.JOINT_GOAL_POSITION, json.dumps(np.asarray(q, float).tolist()))
    time.sleep(settle_s)


def card_plane(arm) -> tuple[np.ndarray, np.ndarray]:
    """The card's pose (arm base) from the wrist camera's view of it now: (4x4, the arm's joints then)."""
    from chefbot.interface.markers import base_card_points
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera
    from chefbot.onboard.calibration.joint import _pnp, corners
    from chefbot.onboard.calibration.wrist import wrist_mount
    from chefbot.onboard.calibration.wrist_base import _detector

    pts = base_card_points()
    with Camera(WRIST_D405) as cam:
        c = corners(cam, _detector(), pts, need=len(pts))
        if c is None:
            raise SystemExit("the wrist camera doesn't see both of the card's markers (ids 1, 2): wrist_base look")
        q = arm.q()
        return fr3.flange(q) @ wrist_mount() @ _pnp(c, pts, cam.K, cam.D), q


def counter_height(arm, spot_xy, guess_z: float, heading_x, confirm: bool = False) -> float | None:
    """The counter's height at spot_xy from the wrist camera's depth, the camera LOOK_FROM straight above it: a plane
    least-squares fitted to its points within 3 cm of the spot (arm base frame), evaluated at the spot."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera
    from chefbot.onboard.calibration.wrist import ik, wrist_mount

    X = wrist_mount()
    zc = np.array([0.0, 0.0, -1.0])  # the camera's view axis (OpenCV z) down
    xc = heading_x - (heading_x @ zc) * zc
    xc /= np.linalg.norm(xc)
    cam_goal = np.eye(4)
    cam_goal[:3, :3] = np.column_stack([xc, np.cross(zc, xc), zc])
    cam_goal[:3, 3] = [spot_xy[0], spot_xy[1], guess_z + LOOK_FROM]
    q = ik(arm.q(), cam_goal @ np.linalg.inv(X))
    if q is None:
        print("  no joints put the camera straight above the spot")
        return None
    arm.move_to(q, "the wrist camera straight above the counter's spot", confirm=confirm)
    arm.still(0.5)
    with Camera(WRIST_D405) as cam:
        zs = [cam.frame()["depth"] for _ in range(10)]
        K = cam.K
    d = np.median(np.stack(zs), axis=0)
    v, u = np.nonzero(d > 0.05)
    pts_cam = np.column_stack([(u - K[0, 2]) * d[v, u] / K[0, 0], (v - K[1, 2]) * d[v, u] / K[1, 1], d[v, u]])
    C = fr3.flange(arm.q()) @ X
    P = pts_cam @ C[:3, :3].T + C[:3, 3]
    near = P[np.linalg.norm(P[:, :2] - spot_xy, axis=1) < 0.03]
    if len(near) < 200:
        print(f"  only {len(near)} depth points around the spot")
        return None
    A = np.column_stack([near[:, 0] - spot_xy[0], near[:, 1] - spot_xy[1], np.ones(len(near))])
    coef, *_ = np.linalg.lstsq(A, near[:, 2], rcond=None)
    resid = near[:, 2] - A @ coef
    keep = np.abs(resid) < 3 * resid.std() + 1e-4
    coef, *_ = np.linalg.lstsq(A[keep], near[keep, 2], rcond=None)
    tilt = np.degrees(np.arctan(np.hypot(coef[0], coef[1])))
    print(f"  the counter at {coef[2]:.4f} m there (depth, {keep.sum()} points, {np.std(near[keep, 2] - A[keep] @ coef) * 1000:.2f} mm rms, "
          f"tilted {tilt:.1f} deg), {d[v, u].mean():.3f} m from the camera")  # fmt: skip
    return float(coef[2])


def touch(touches: int = 3, on: str = "counter") -> dict:
    from chefbot.interface.workspace import BODY
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import Arm, Stop, set_table
    from chefbot.onboard.calibration.wrist import ik

    arm = Arm()
    card, _ = card_plane(arm)
    n, o = card[:3, 2], card[:3, 3]
    if n[2] < 0.95:
        raise SystemExit(f"the card isn't lying flat (its normal {np.round(n, 3).tolist()}): lay it flat on the counter")
    hand = BODY.hand_m
    print(f"the card at {np.round(o, 4).tolist()} m (arm base), tilted {np.degrees(np.arccos(n[2])):.1f} deg; "
          f"the fingertips expected at {hand:.4f} m from the flange (robot.yaml hand_m)")  # fmt: skip
    set_table(None)  # the hand goes onto the card on purpose; every other move is the planner's as usual
    F0 = fr3.flange(arm.q())
    z = np.array([0.0, 0.0, -1.0])
    x = F0[:3, 0] - (F0[:3, 0] @ z) * z
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])  # straight down, the heading the hand has now
    spots = [np.zeros(2), np.array([0.01, 0.01]), np.array([-0.01, -0.01]), np.array([0.01, -0.01])][:touches]
    toward = -o[:2] / np.linalg.norm(o[:2])  # from the card toward the arm base
    out = []
    try:
        for k, d in enumerate(spots):
            if on == "counter":
                xy = o[:2] + toward * IN_FRONT + d
                plane_z = counter_height(arm, xy, o[2], x, confirm=k == 0)
                if plane_z is None:
                    continue
            else:
                xy = o[:2] + d
                plane_z = o[2] - (n[0] * d[0] + n[1] * d[1]) / n[2]  # the card's surface under that spot
            start = np.eye(4)
            start[:3, :3], start[:3, 3] = R, [xy[0], xy[1], plane_z + hand + ABOVE]
            q = ik(arm.q(), start)
            if q is None:
                raise SystemExit(f"touch {k + 1}: no joints put the hand over the card there")
            arm.move_to(q, f"touch {k + 1}/{len(spots)}: the hand {ABOVE * 100:.0f} cm above where the fingertips should meet the {on}",
                        confirm=k == 0 and on == "card")  # fmt: skip
            arm.still(1.0)
            base = _ext(arm)
            contact, first, trace = None, None, []
            for i in range(1, int((ABOVE + MAX_BELOW) / STEP) + 1):
                T = start.copy()
                T[2, 3] -= i * STEP
                qi = ik(q, T)
                if qi is None:
                    break
                _goto(arm, qi)
                q = qi
                change = np.abs(_ext(arm) - base)[[1, 3, 5]]
                trace.append(f"{change.max():.2f}")
                if change.max() > CONTACT_NM:
                    if first is None:  # the first step over: where the fingertips met it, if the next one agrees
                        first = (i, fr3.flange(arm.q())[2, 3], change)
                        continue
                    contact = first[1]
                    print(f"  contact {first[0]} mm down: torque change {np.round(first[2], 2).tolist()} N m (joints 2, 4, 6), "
                          f"then {np.round(change, 2).tolist()}")  # fmt: skip
                    break
                first = None  # a one-step blip: not contact
            print(f"  each step's largest torque change (N m): {' '.join(trace)}")
            up = ik(q, start)
            _goto(arm, q if up is None else up, 1.5)  # back up
            if contact is None:
                print(f"  touch {k + 1}: no contact within {MAX_BELOW * 100:.0f} cm below the expected one: skipped")
                continue
            length = contact - plane_z
            out.append(length)
            print(f"  touch {k + 1}: the flange {contact:.4f} m at contact, the {on} {plane_z:.4f}: fingertips {length:.4f} m from the flange")
    except (Stop, KeyboardInterrupt):
        arm.hold()
        print("stopped: holding")
    if not out:
        return {"touches": 0}
    res = {"touches": len(out), "fingertips_from_flange_m": round(float(np.mean(out)), 4),
           "spread_mm": round(float(np.ptp(out)) * 1000, 1), "robot_yaml_hand_m": hand,
           "difference_mm": round((float(np.mean(out)) - hand) * 1000, 1)}  # fmt: skip
    print(json.dumps(res, indent=1))
    print("(the contact is found a step late: the true length is up to ~1 mm more than this, the press of the pads aside)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("touch")
    t.add_argument("--touches", type=int, default=3)
    t.add_argument("--on", choices=("counter", "card"), default="counter")
    a = ap.parse_args()
    touch(a.touches, a.on)


if __name__ == "__main__":
    main()
