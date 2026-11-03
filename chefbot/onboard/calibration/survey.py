"""Look at places on the base with the wrist camera, to pick where a calibration board could live.

    python -m chefbot.onboard.calibration.survey [--spots pole front-left ...] [--at X Y Z] [--size 0.070]

For each spot (the arm base frame: x forward, y left, z up from the arm's mounting plate): the camera ~0.35 m from
it, as close to looking straight down as the arm reaches clear of the keep-out (by the pole, from the side), Enter,
a frame (saved), the markers it sees (the one of --size: where it is, and how many of 60 calibration views of a
board there the arm can reach), Enter, back to the start. Every move's path stays out of the keep-out and 5 cm above
the plate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration.wrist import DICTIONARY, MARKER_ID, _T, _object_points, _rot, ik, wrist_mount

SPOTS = {  # on the plate (z 0): beside the arm toward the front, at its side, behind it, by the pole
    "front-left": (0.20, 0.20, 0.0),
    "front-right": (0.20, -0.20, 0.0),
    "right": (0.0, -0.25, 0.0),
    "behind-right": (-0.20, -0.20, 0.0),
    "pole": (-0.10, 0.20, 0.0),
}
DISTANCE = 0.35
MAX_ROLL = np.radians(100)  # the wrist's turn from where it starts, at most (the camera's cable)
PLATE_CLEAR = 0.05  # m: every arm point (bar the base's own) at least this above the plate
SEEDS = (np.array([0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785]), np.array([1.5, -0.5, 0.0, -2.4, 0.0, 2.0, 0.785]),
         np.array([-1.5, -0.5, 0.0, -2.4, 0.0, 2.0, 0.785]), np.array([2.4, -0.6, 0.0, -2.3, 0.0, 1.8, 0.785]),
         np.array([-2.4, -0.6, 0.0, -2.3, 0.0, 1.8, 0.785]))  # fmt: skip


CAMERA_MARGIN = 0.05  # m: the wrist camera (off the flange, outside the arm's own points) kept this far off the keep-out


def clear(q) -> bool:
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import _inside, arm_points, in_keep_out

    camera = (fr3.flange(q) @ wrist_mount())[:3, 3]
    return not in_keep_out(q) and not _inside(camera[None], CAMERA_MARGIN)[0] and arm_points(q)[2:, 2].min() > PLATE_CLEAR


def path_ok(q0, q1, samples: int = 30) -> bool:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    return all(clear(q0 + (q1 - q0) * t) for t in np.linspace(0, 1, samples))


def _view(spot, normal, tilt_deg, az_deg, dist):
    """The camera (OpenCV axes) `dist` from spot, its view axis tilted tilt_deg off -normal toward azimuth az_deg."""
    a = np.cross(normal, [1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    t, z = np.radians(tilt_deg), np.radians(az_deg)
    v = np.cos(t) * normal + np.sin(t) * (np.cos(z) * a + np.sin(z) * b)  # from the spot toward the camera
    zc = -v
    xc = np.cross(zc, [0, 0, 1.0])
    xc = xc / np.linalg.norm(xc) if np.linalg.norm(xc) > 1e-3 else np.array([1.0, 0, 0])
    return _T(np.stack([xc, np.cross(zc, xc), zc], 1), np.asarray(spot) + v * dist)


def look_at(spot, q_from, normal=(0, 0, 1.0), dist: float = DISTANCE) -> np.ndarray | None:
    """Joints with the camera looking at spot, as straight on as reachable, the path from q_from clear."""
    X_inv = np.linalg.inv(wrist_mount())
    for tilt in range(0, 50, 5):
        for az in range(0, 360, 45):
            for roll in (0, 90, 180, -90):
                Tf = _view(spot, np.asarray(normal, float), tilt, az, dist) @ _T(_rot([0, 0, 1], roll), [0, 0, 0]) @ X_inv
                for seed in (q_from, *SEEDS):
                    q = ik(seed, Tf)
                    if q is not None and abs(q[6] - q_from[6]) < MAX_ROLL and clear(q) and path_ok(q_from, q):
                        return q
            if tilt == 0:
                break  # straight on: every azimuth is the same view
    return None


def reachable_views(spot, normal=(0, 0, 1.0), n: int = 60, seed: int = 0) -> int:
    """Of n calibration views of a board at spot (0.25-0.4 m, up to 25 deg off straight on), how many reach clear."""
    rng = np.random.default_rng(seed)
    X_inv = np.linalg.inv(wrist_mount())
    ok = 0
    for _ in range(n):
        Tf = _view(spot, np.asarray(normal, float), rng.uniform(0, 25), rng.uniform(0, 360), rng.choice((0.25, 0.32, 0.4)))
        Tf = Tf @ _T(_rot([0, 0, 1], rng.uniform(-180, 180)), [0, 0, 0]) @ X_inv
        ok += any((q := ik(s, Tf)) is not None and clear(q) for s in SEEDS)
    return ok


def survey(spots: dict, size: float, out_dir: Path) -> None:
    import cv2

    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import Arm, Stop
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera

    arm = Arm()
    q_start = arm.q()
    X = wrist_mount()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)), cv2.aruco.DetectorParameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, spot in spots.items():
            print(f"\n== {name}: {spot}")
            q = look_at(spot, q_start)
            if q is None:
                print("  the arm can't look at it from anywhere clear: skipped")
                continue
            arm.move_to(q, f"{name}: the camera {DISTANCE} m from it")
            with Camera(WRIST_D405) as cam:
                f = cam.frame()
            q_now = arm.q()
            cv2.imwrite(str(out_dir / f"survey_{name}.png"), f["bgr"])
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(f["bgr"], cv2.COLOR_BGR2GRAY))
            seen = [] if ids is None else ids.ravel().tolist()
            print(f"  image: {out_dir / f'survey_{name}.png'}; {DICTIONARY} markers seen: {seen or 'none'}")
            if MARKER_ID in seen:
                c = corners[seen.index(MARKER_ID)][0]
                ok, rvec, tvec = cv2.solvePnP(_object_points(size), c, f["K"], f["D"], flags=cv2.SOLVEPNP_IPPE_SQUARE)
                M = fr3.flange(q_now) @ X @ _T(cv2.Rodrigues(rvec)[0], tvec)
                n = M[:3, 2] if M[2, 2] > 0 else -M[:3, 2]
                print(f"  marker {MARKER_ID} at {np.round(M[:3, 3], 3).tolist()} m, face up by {np.degrees(np.arccos(min(1, n[2]))):.0f} deg off")
                print(f"  a board there: {reachable_views(M[:3, 3], n)}/60 calibration views reachable (about 40 is good)")
            arm.move_to(q_start, f"{name}: back to the start")
    except (Stop, KeyboardInterrupt):
        arm.hold()
        print("stopped: holding")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spots", nargs="+", choices=sorted(SPOTS), default=["pole", "front-left", "front-right", "right", "behind-right"])
    ap.add_argument("--at", type=float, nargs=3, default=None, help="look here instead (arm base frame, m)")
    ap.add_argument("--size", type=float, default=0.070, help="the loose marker's black square (m)")
    ap.add_argument("--out", type=Path, default=Path.home() / "chefbot_calib")
    args = ap.parse_args()
    spots = {"here": tuple(args.at)} if args.at else {k: SPOTS[k] for k in args.spots}
    survey(spots, args.size, args.out)


if __name__ == "__main__":
    main()
