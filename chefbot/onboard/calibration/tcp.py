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
watched against where it started as the force at the fingertips; past contact it presses to PRESS_N and the force
against the height is fitted (zero force: the contact), then the hand goes back up. It never goes
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
CONTACT_N = 4.0  # N: the force at the fingertips (from all the joints' external torque) that counts as contact, twice
PRESS_N = 25.0  # N: past contact it presses on, in PRESS_STEP steps, to this (or PRESS_MAX further), then the force
PRESS_STEP = 0.0005  # m      against the flange's height is fitted with a line: where it is zero is where the
PRESS_MAX = 0.008  # m        fingertips met the surface (no threshold, nothing of how hard it pressed in it)
SETTLE_S = 0.4  # s after each step before the torques are read
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


def _clear(q) -> bool:
    """Out of the pole's keep-out (an IK solution may put the elbow under the camera's clamp)."""
    from chefbot.interface.workspace import in_keep_out

    return not in_keep_out(np.asarray(q, float))


def _look_goal(spot_xy, z, heading_x) -> np.ndarray:
    """The flange pose putting the wrist camera LOOK_FROM straight above (spot_xy, z), looking down."""
    from chefbot.onboard.calibration.wrist import wrist_mount

    zc = np.array([0.0, 0.0, -1.0])
    xc = heading_x - (heading_x @ zc) * zc
    xc /= np.linalg.norm(xc)
    cam_goal = np.eye(4)
    cam_goal[:3, :3] = np.column_stack([xc, np.cross(zc, xc), zc])
    cam_goal[:3, 3] = [spot_xy[0], spot_xy[1], z + LOOK_FROM]
    return cam_goal @ np.linalg.inv(wrist_mount())


def _force(q, delta_tau, tips_m: float) -> float:
    """The force straight up at the fingertips (N) that explains a change in the joints' external torque: the wrench
    there by least squares through the arm's Jacobian at the fingertips, its z."""
    from chefbot.onboard.calibration import fr3

    F = fr3.flange(q)
    J = fr3.jacobian(q)
    r = F[:3, 2] * tips_m  # flange to fingertips
    Jv = J[:3] - np.cross(r[None, :], J[3:].T).T  # the fingertip point's linear rows: v_tip = v + w x r
    Jt = np.vstack([Jv, J[3:]])
    w, *_ = np.linalg.lstsq(Jt.T, np.asarray(delta_tau, float), rcond=None)
    return float(w[2])


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


def counter_height(arm, spot_xy, guess_z: float, heading_x, confirm: bool = False) -> tuple[float, np.ndarray] | None:
    """The counter's height at spot_xy from the wrist camera's depth, the camera LOOK_FROM straight above it: a plane
    least-squares fitted to its points within 3 cm of the spot (arm base frame), evaluated at the spot."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera
    from chefbot.onboard.calibration.wrist import ik_any, wrist_mount

    X = wrist_mount()
    q = ik_any(_look_goal(spot_xy, guess_z, heading_x), arm.q(), _clear)
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
    normal = np.array([-coef[0], -coef[1], 1.0])
    return float(coef[2]), normal / np.linalg.norm(normal)


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
    from chefbot.onboard.calibration.wrist import ik_any

    where = np.zeros(2)
    if on == "counter":  # a spot on the counter beside the card the arm can look at and reach clear of the keep-out
        side = np.array([-toward[1], toward[0]])
        for name, dirn in (("in front of", toward), ("beyond", -toward), ("left of", side), ("right of", -side)):
            xy = o[:2] + dirn * IN_FRONT
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, [xy[0], xy[1], o[2] + hand + ABOVE]
            if ik_any(_look_goal(xy, o[2], x), arm.q(), _clear) is not None and ik_any(T, arm.q(), _clear) is not None:
                where = dirn * IN_FRONT
                print(f"the counter {IN_FRONT * 100:.0f} cm {name} the card")
                break
        else:
            raise SystemExit("no spot beside the card the arm can reach clear of the keep-out: move the card")
    out = []
    try:
        for k, d in enumerate(spots):
            if on == "counter":
                xy = o[:2] + where + d
                got = counter_height(arm, xy, o[2], x, confirm=k == 0)
                if got is None:
                    continue
                plane_z, up = got
            else:
                xy = o[:2] + d
                plane_z = o[2] - (n[0] * d[0] + n[1] * d[1]) / n[2]  # the card's surface under that spot
                up = n
            # The hand square to the surface (its measured normal), not to the arm's vertical: the fingertips flat on it
            zs = -up
            xs = x - (x @ zs) * zs
            xs /= np.linalg.norm(xs)
            Rs = np.column_stack([xs, np.cross(zs, xs), zs])
            start = np.eye(4)
            start[:3, :3], start[:3, 3] = Rs, [xy[0], xy[1], plane_z + hand + ABOVE]
            q = ik_any(start, arm.q(), _clear)
            if q is None:
                raise SystemExit(f"touch {k + 1}: no joints put the hand over the card there")
            arm.move_to(q, f"touch {k + 1}/{len(spots)}: the hand {ABOVE * 100:.0f} cm above where the fingertips should meet the {on}",
                        confirm=k == 0 and on == "card")  # fmt: skip
            arm.still(1.0)
            base = _ext(arm)
            contact, first, trace, rows = None, None, [], []
            sign = None  # the force's sign as these torques give it (the surface pushes up)
            depth, i = 0.0, 0
            while depth < ABOVE + MAX_BELOW:
                step = STEP if first is None else PRESS_STEP
                depth += step
                i += 1
                T = start.copy()
                T[:3, 3] -= depth * up  # toward the surface, along its normal
                qi = ik(q, T)
                if qi is None or not _clear(qi):
                    break
                _goto(arm, qi)
                q = qi
                qm = arm.q()
                f_raw = _force(qm, _ext(arm) - base, hand)
                f = abs(f_raw) if sign is None else sign * f_raw
                Fm = fr3.flange(qm)
                h = float((Fm[:3, 3] - np.r_[xy, plane_z]) @ up)  # the flange's height over the surface, measured
                rows.append((h, f))
                trace.append(f"{f:.1f}")
                if first is None:
                    if f > CONTACT_N:
                        if len(rows) >= 2 and rows[-2][1] > CONTACT_N:  # two in a row: contact, press on from here
                            sign = np.sign(f_raw)
                            first = (len(rows) - 2, depth, float(np.degrees(np.arccos(np.clip(-Fm[:3, 2] @ up, -1, 1)))))
                            print(f"  contact about {(depth - step) * 1000:.0f} mm down ({f:.1f} N); pressing to {PRESS_N:.0f} N")
                    continue
                if f >= PRESS_N or depth - first[1] >= PRESS_MAX:
                    break
            print(f"  each step's force at the fingertips (N): {' '.join(trace)}")
            if first is not None:
                pts = np.array(rows[max(0, first[0] - 1):])
                fit = pts[(pts[:, 1] > CONTACT_N * 0.5) & (pts[:, 1] <= PRESS_N * 1.2)]
                if len(fit) >= 3:
                    slope, icpt = np.polyfit(fit[:, 1], fit[:, 0], 1)  # height = icpt + slope * force
                    contact = float(icpt) + plane_z
                    stiff = -1.0 / slope / 1000 if slope else float("nan")
                    print(f"  fitted on {len(fit)} steps: zero force at the flange {icpt * 1000:.1f} mm over the surface "
                          f"(the pressing {stiff:.1f} N/mm); the hand {first[2]:.2f} deg off square")  # fmt: skip
                else:
                    contact = float(pts[0, 0]) + plane_z
                    print(f"  only {len(fit)} steps to fit: the first contact's height")
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
