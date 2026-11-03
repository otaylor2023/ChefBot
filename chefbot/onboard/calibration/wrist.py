"""The wrist D405's pose on the flange (eye-in-hand), from an ArUco / ChArUco board lying still in front of the robot.

    python -m chefbot.onboard.calibration.wrist look [--pose ready|forward]  # moves the arm (Enter first), then says
                                                                             # which markers the camera sees (look.png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Looking at the board: ready (the hand pointing down, ~0.5 m ahead of the arm base), or tipped forward from it.
LOOK_POSES = {
    "ready": [0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785],
    "forward": [0.0, -0.45, 0.0, -2.0, 0.0, 1.95, 0.785],
}


def detect_markers(gray) -> list[tuple[str, np.ndarray, list]]:
    """Every predefined ArUco dictionary that finds at least two markers: (name, ids, corners), most markers first."""
    import cv2

    found = []
    for name in (n for n in dir(cv2.aruco) if n.startswith("DICT_")):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
        if ids is not None and len(ids) >= 2:
            found.append((name, ids.flatten(), corners))
    return sorted(found, key=lambda f: -len(f[1]))


def look(pose: str, out: Path) -> None:
    import cv2

    from chefbot.onboard.calibration.arm import Arm
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera

    Arm().move_to(np.array(LOOK_POSES[pose]), f"look pose: {pose}")
    with Camera(WRIST_D405) as cam:
        f = cam.frame()
    gray = cv2.cvtColor(f["bgr"], cv2.COLOR_BGR2GRAY)
    found = detect_markers(gray)
    img = f["bgr"].copy()
    if found:
        name, ids, corners = found[0]
        cv2.aruco.drawDetectedMarkers(img, corners, ids.reshape(-1, 1))
        for n, i, _ in found[:4]:
            print(f"  {n}: {len(i)} markers, ids {sorted(i.tolist())[:30]}")
        z = f["depth"][f["depth"] > 0]
        print(f"  depth: median {np.median(z):.3f} m" if z.size else "  no depth")
    else:
        print("  no markers seen: move the board into view (or try --pose forward)")
    cv2.imwrite(str(out), img)
    print(f"  image: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("look")
    a.add_argument("--pose", choices=sorted(LOOK_POSES), default="ready")
    a.add_argument("--out", type=Path, default=Path("look.png"))
    args = ap.parse_args()
    if args.cmd == "look":
        look(args.pose, args.out)


if __name__ == "__main__":
    main()
