"""Check the camera calibrations on the arm: find the marker with a camera, then put the fingertips an inch from it.

    python -m chefbot.onboard.calibration.check [--cameras wrist pole] [--size 0.070]

For each camera in turn, from where the arm is now: the camera finds the marker (its pose through the calibration in
environments/real/config.yaml and the arm's joints), the hand goes 10 cm out from the marker's face, pointing into
it, then to 1 inch (the fingertips), then back the same way to the start. Enter before every move. With both cameras
it also prints how far apart their two estimates of the marker are: both calibrations agreeing.

Where the marker can be (2026-11-03): flat on the chair seat, for the wrist camera from the ready pose (wrist look
puts the arm there); the pole camera looks nearly level from about the hand's height and doesn't see the seat. For
the pole camera: on the chair's back, leaning back about 30 deg, facing the robot (the hand can reach it there;
upright, or low and tilted, it can't).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration.wrist import DICTIONARY, MARKER_ID, _object_points, _T

INCH = 0.0254
# The fingers' ends, past the hand's TCP (the pads' middle, arm.TCP_Z): ChefBot's hand model has the pads 0.094-0.111 m
# out from the flange, their middle 0.103 (embodiments/tidybot_fr3/robot.yaml hand)
FINGER_END_PAST_TCP = 0.111 - 0.103
APPROACH = 0.10  # m out from the marker, first
# IK starts besides where the arm is: reaching low and out; the hand pointing forward
SEEDS = (np.array([0.0, 0.2, 0.0, -2.3, 0.0, 2.5, 0.785]), np.array([0.0, -0.3, 0.0, -1.9, 0.0, 3.1, 0.785]))


def camera_mounts() -> dict:
    """The cameras in environments/real/config.yaml: (parent, T), the pole's moved into the arm base frame."""
    import yaml

    from chefbot.onboard.calibration.pole import _arm_mount

    cams = yaml.safe_load((Path(__file__).parents[2] / "environments" / "real" / "config.yaml").read_text())["cameras"]
    out = {}
    for name, c in cams.items():
        T = np.array(c["T"], float)
        out[name] = ("flange", T) if c["parent"] == "flange" else ("armbase", np.linalg.inv(_arm_mount()) @ T)
    return out


def find_marker(name: str, q, dictionary: str, marker_id: int, size: float):
    """The marker's pose in the arm base, from camera `name` with the arm at q; None if it isn't seen."""
    import cv2

    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.camera import POLE_D455, WRIST_D405, Camera
    from chefbot.onboard.calibration.wrist import corners_now

    parent, T = camera_mounts()[name]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), cv2.aruco.DetectorParameters())
    with Camera({"wrist": WRIST_D405, "pole": POLE_D455}[name]) as cam:
        c = corners_now(cam, det, marker_id)
        if c is None:
            return None
        ok, rvec, tvec = cv2.solvePnP(_object_points(size), c, cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    T_cam = (fr3.flange(q) @ T) if parent == "flange" else T
    M = T_cam @ _T(cv2.Rodrigues(rvec)[0], tvec)
    if M[:3, 2] @ (T_cam[:3, 3] - M[:3, 3]) < 0:  # its z out of the face, toward the camera
        M[:3, 1:3] *= -1
    return M


def hand_at(M, out: float, q_ref) -> np.ndarray | None:
    """Joints with the fingers' ends `out` m from the marker's center along its face's normal, the hand pointing into
    it, turned as little as possible from q_ref's."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import TCP_Z
    from chefbot.onboard.calibration.wrist import ik

    from chefbot.onboard.calibration.wrist import _rot

    n = M[:3, 2]
    z = -n
    x = fr3.flange(q_ref)[:3, 0]
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-3:
        x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R0 = np.stack([x, np.cross(z, x), z], 1)
    tip = M[:3, 3] + n * (out + FINGER_END_PAST_TCP)  # the TCP, so that the fingers' ends are `out` from it
    best = None
    for roll in (0, 45, -45, 90, -90, 135, -135, 180):  # the hand turned about its pointing axis, least first
        R = R0 @ _rot([0, 0, 1], roll)
        for seed in (q_ref, *SEEDS):
            q = ik(seed, _T(R, tip - z * TCP_Z))
            if q is not None and (best is None or np.abs(q - q_ref).max() < np.abs(best - q_ref).max()):
                best = q
        if best is not None:
            return best
    return None


def check(cameras, dictionary: str, marker_id: int, size: float) -> None:
    from chefbot.onboard.calibration.arm import Arm, Stop

    arm = Arm()
    q_start = arm.q()
    found = {}
    try:
        for name in cameras:
            print(f"\n== {name} camera")
            M = find_marker(name, q_start, dictionary, marker_id, size)
            if M is None:
                print(f"  the {name} camera doesn't see {dictionary} id {marker_id} from here: skipped")
                continue
            found[name] = M
            print(f"  marker at {np.round(M[:3, 3], 3).tolist()} m (arm base), its face toward {np.round(M[:3, 2], 2).tolist()}")
            q_near, q_inch = hand_at(M, APPROACH, q_start), None
            if q_near is not None:
                q_inch = hand_at(M, INCH, q_near)
            if q_near is None or q_inch is None:
                print("  the arm can't reach the hand to it there (IK): skipped")
                continue
            arm.move_to(q_near, f"{name}: 10 cm out from the marker")
            arm.move_to(q_inch, f"{name}: the fingers' ends 1 inch from the marker: look how far off it is")
            arm.move_to(q_near, f"{name}: back out to 10 cm")
            arm.move_to(q_start, f"{name}: back to the start")
    except (Stop, KeyboardInterrupt):
        arm.hold()
        print("stopped: holding")
        return
    if len(found) == 2:
        a, b = found.values()
        d = (a[:3, 3] - b[:3, 3]) * 1000
        print(f"\nthe two cameras' marker positions differ by {np.linalg.norm(d):.1f} mm ({np.round(d, 1).tolist()}, wrist minus pole)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", choices=["wrist", "pole"], default=["wrist", "pole"])
    ap.add_argument("--dict", default=DICTIONARY)
    ap.add_argument("--id", type=int, default=MARKER_ID)
    ap.add_argument("--size", type=float, default=0.070, help="the marker's black square (m)")
    args = ap.parse_args()
    check(args.cameras, args.dict, args.id, args.size)


if __name__ == "__main__":
    main()
