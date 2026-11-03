"""The pole D455's pose on the base (eye-to-hand), from an ArUco marker fixed flat and stiff to the hand.

    python -m chefbot.onboard.calibration.pole look                    # what the pole camera sees now (no motion)
    python -m chefbot.onboard.calibration.pole record out.json --dict DICT_4X4_50 --id 0 --size 0.127
        # from a pose where the camera sees the marker: moves the hand around it (Enter once), corners per pose
    python -m chefbot.onboard.calibration.pole solve out.json [--skip 3 7]

Start with the marker facing the pole camera: put the arm there by hand (Desk's guiding mode, or the arm driver
alone, floating, before OpenSai), then start OpenSai (it holds where the arm is) and record. The moves stay out of
the space behind the arm on its left (the pole) with extra room for the marker sticking out of the hand.

The camera comes out in the arm base and in ChefBot's base frame (the floor under the base center: the arm base is
config/robot_tidybot_fr3.yaml's arm_mount above it), OpenCV axes (x right, y down, z forward), as
environments/real/config.yaml's cameras take it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration.wrist import DICTIONARY, FRAMES, MARKER_ID, _object_points, _pnp, _refine, _rot, _T, ax_xb

MARKER_MARGIN = 0.15  # m: the keep-out grown by this (the marker sticks out of the hand)
SHIFTS_M = ((0, 0, 0), (0.08, 0, 0), (-0.08, 0, 0), (0, 0.08, 0), (0, -0.08, 0), (0, 0, 0.08), (0, 0, -0.08))
TURNS_DEG = (([0, 0, 1], 0), ([1, 0, 0], 18), ([1, 0, 0], -18), ([0, 1, 0], 18), ([0, 1, 0], -18), ([0, 0, 1], 25), ([0, 0, 1], -25))


def _detector(dictionary: str):
    import cv2

    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), cv2.aruco.DetectorParameters())


def look(out: Path) -> None:
    import cv2

    from chefbot.onboard.calibration.camera import POLE_D455, Camera
    from chefbot.onboard.calibration.wrist import detect_markers

    with Camera(POLE_D455) as cam:
        f = cam.frame()
    found = detect_markers(cv2.cvtColor(f["bgr"], cv2.COLOR_BGR2GRAY))
    img = f["bgr"].copy()
    if found:
        cv2.aruco.drawDetectedMarkers(img, found[0][2], found[0][1].reshape(-1, 1))
        for n, ids, corners in found[:4]:
            side = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
            print(f"  {n}: ids {sorted(ids.tolist())[:20]} ({side:.0f} px across)")
    else:
        print("  no markers seen: turn the marker toward the pole camera")
    cv2.imwrite(str(out), img)
    print(f"  image: {out}")


def poses_around(q_look) -> list[np.ndarray]:
    """The hand shifted and turned about the fingertips (near the marker), from the look pose: shifts paired with
    turns so that each pose differs in both."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import TCP_Z, path_clear
    from chefbot.onboard.calibration.wrist import ik

    T0 = fr3.flange(q_look)
    pivot = T0[:3, 3] + T0[:3, 2] * TCP_Z
    poses = []
    for k, shift in enumerate(SHIFTS_M):
        found = 0
        for n in range(len(TURNS_DEG)):  # the first two turns that work, starting from a different one each shift
            axis, deg = TURNS_DEG[(k + 2 * n) % len(TURNS_DEG)]
            if found == 2 or (not any(shift) and deg == 0):  # (the look pose itself is already the first)
                continue
            R = _rot(axis, deg)
            T = np.eye(4)
            T[:3, :3] = R @ T0[:3, :3]
            T[:3, 3] = pivot + np.asarray(shift) + R @ (T0[:3, 3] - pivot)
            q = ik(q_look, T)
            if q is not None and np.abs(q - q_look).max() < np.radians(90) and path_clear(q_look, q, margin=MARKER_MARGIN):
                poses.append(q)
                found += 1
    out, prev = [], np.asarray(q_look, float)
    for q in poses:  # each move from the one before clear too, else by way of the look pose
        if not path_clear(prev, q, margin=MARKER_MARGIN):
            out.append(np.asarray(q_look, float))
        out.append(q)
        prev = q
    return out


def record(out: Path, dictionary: str, marker_id: int, size: float | None) -> None:
    from chefbot.onboard.calibration.arm import Arm, Stop, in_keep_out
    from chefbot.onboard.calibration.camera import POLE_D455, Camera
    from chefbot.onboard.calibration.wrist import corners_now

    arm, det = Arm(), _detector(dictionary)
    marker = {"dictionary": dictionary, "id": marker_id, "size_m": size}
    samples = []
    with Camera(POLE_D455) as cam:
        q_look = arm.q()
        if in_keep_out(q_look, MARKER_MARGIN):
            raise SystemExit("the hand (with the marker) is too near the pole: start further from it")
        if corners_now(cam, det, marker_id) is None:
            raise SystemExit(f"the pole camera doesn't see {dictionary} id {marker_id}: run look, turn the marker to it")
        poses = [q_look] + poses_around(q_look)
        print(f"{len(poses)} poses")
        save = lambda: out.write_text(json.dumps({"K": cam.K.tolist(), "D": cam.D.tolist(), "marker": marker, "samples": samples}, indent=1))
        try:
            for i, q in enumerate(poses):
                arm.move_to(q, f"pose {i + 1}/{len(poses)}", confirm=i == 1, margin=MARKER_MARGIN)
                arm.still(0.5)
                c = corners_now(cam, det, marker_id, FRAMES)
                if c is None:
                    print("  marker not seen: skipped")
                    continue
                samples.append({"q": arm.still(0.5)["q"], "corners": c.tolist()})
                print(f"  recorded ({len(samples)} so far)")
                save()
        except (Stop, KeyboardInterrupt):
            arm.hold()
            print("stopped: holding")
        finally:
            save()
            print(f"{len(samples)} poses in {out}")
        arm.move_to(q_look, "back to the start", confirm=False, margin=MARKER_MARGIN)


def _pole_chain(g, Y, Z):
    return np.linalg.inv(Y) @ g @ Z  # the marker in the camera: Y the camera in the arm base, Z the marker on the flange


def _arm_mount() -> np.ndarray:
    """The arm base in ChefBot's base frame (config/robot_tidybot_fr3.yaml's arm_mount)."""
    import yaml

    m = yaml.safe_load((Path(__file__).parents[2] / "config" / "robot_tidybot_fr3.yaml").read_text())["arm_mount"]
    return _T(_rot([0, 0, 1], m.get("yaw_deg", 0.0)), m["xyz"])


def solve(path: Path, skip=(), size: float | None = None) -> dict:
    from chefbot.onboard.calibration import fr3

    d = json.loads(path.read_text())
    K, D = np.array(d["K"]), np.array(d["D"])
    size = size or d.get("marker", {}).get("size_m")
    if size is None:
        raise SystemExit("the marker's size isn't known: pass --size (its black square, m)")
    samples = [s for i, s in enumerate(d["samples"]) if i + 1 not in set(skip)]
    if len(samples) < 5:
        raise SystemExit(f"{len(samples)} poses: need at least 5")
    G = [fr3.flange(s["q"]) for s in samples]
    C = _pnp(samples, K, D, size)
    # G_i Z = Y C_i for every pose: (G_j G_i^-1) Y = Y (C_j C_i^-1)
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    Y0 = ax_xb([G[j] @ np.linalg.inv(G[i]) for i, j in pairs], [C[j] @ np.linalg.inv(C[i]) for i, j in pairs])
    Zs = [np.linalg.inv(g) @ Y0 @ c for g, c in zip(G, C)]
    Z0 = _T(Zs[0][:3, :3], np.mean([z[:3, 3] for z in Zs], 0))
    Y, Z, _, res = _refine(samples, K, D, Y0, Z0, size, fit_size=False, chain=_pole_chain)
    # the marker on the flange from each pose on its own (through Y): it is fixed there, so this is the error
    on_flange = np.array([(np.linalg.inv(g) @ Y @ c)[:3, 3] for g, c in zip(G, C)])
    px = np.linalg.norm(res, axis=2)
    T_base = _arm_mount() @ Y
    out = {"poses": len(samples), "skipped": sorted(skip), "marker_size_m": size,
           "T_base_camera": np.round(T_base, 5).tolist(), "T_armbase_camera": np.round(Y, 5).tolist(),
           "camera_in_base_m": np.round(T_base[:3, 3], 4).tolist(),
           "camera_looks_down_deg": round(float(np.degrees(np.arcsin(-T_base[2, 2]))), 1),
           "reprojection_px": {"rms": round(float(np.sqrt(np.mean(px**2))), 2), "max": round(float(px.max()), 2)},
           "reprojection_rms_px_per_pose": np.round(np.sqrt(np.mean(px**2, 1)), 2).tolist(),
           "marker_on_flange_m": np.round(Z[:3, 3], 4).tolist(),
           "marker_per_pose_spread_mm": np.round(np.linalg.norm(on_flange - Z[:3, 3], axis=1) * 1000, 1).tolist()}  # fmt: skip
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("look")
    a.add_argument("--out", type=Path, default=Path("pole_look.png"))
    b = sub.add_parser("record")
    b.add_argument("out", type=Path)
    b.add_argument("--dict", default=DICTIONARY)
    b.add_argument("--id", type=int, default=MARKER_ID)
    b.add_argument("--size", type=float, default=None, help="the marker's black square, measured (m)")
    c = sub.add_parser("solve")
    c.add_argument("data", type=Path)
    c.add_argument("--skip", type=int, nargs="*", default=[])
    c.add_argument("--size", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "look":
        look(args.out)
    elif args.cmd == "record":
        record(args.out, args.dict, args.id, args.size)
    else:
        solve(args.data, args.skip, args.size)


if __name__ == "__main__":
    main()
