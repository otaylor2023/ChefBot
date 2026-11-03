"""The pole D455's pose on the base (eye-to-hand), from an ArUco marker fixed flat and stiff to the hand.

    python -m chefbot.onboard.calibration.pole look                    # what the pole camera sees now (no motion)
    python -m chefbot.onboard.calibration.pole record out.json --card        # the hand card (targets.py: ids 1, 2)
    python -m chefbot.onboard.calibration.pole record out.json --dict DICT_4X4_50 --id 0 --size 0.070  # one marker
        # from a pose where the camera sees it: moves the hand around (Enter once), corners per pose
    python -m chefbot.onboard.calibration.pole solve out.json [--skip 3 7]
        # the camera, and the card's pose on the flange (T_flange_card: what the mount tracker needs)

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
# The hand moved this far from the start (base frame, m), spread wide and low so the fit holds over the workspace
# (2026-11-03: poses only +-8 cm around one spot left the camera's tilt loose: a marker 40 cm lower came out low)
SHIFTS_M = ((0, 0, 0), (0.08, 0, 0), (-0.08, 0, 0), (0, 0.08, 0), (0, -0.08, 0), (0, 0, 0.08), (0, 0, -0.08),
            (0.15, 0, 0), (0, 0.15, 0), (0, -0.15, 0), (0, 0, -0.15), (0.1, 0.1, -0.2), (0.1, -0.1, -0.2), (0.15, 0, -0.25))  # fmt: skip
TURNS_DEG = (([0, 0, 1], 0), ([1, 0, 0], 18), ([1, 0, 0], -18), ([0, 1, 0], 18), ([0, 1, 0], -18), ([0, 0, 1], 25), ([0, 0, 1], -25))


def _detector(dictionary: str):
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)


def card_corners(cam, det, frames: int = FRAMES) -> dict[int, np.ndarray] | None:
    """The hand card's markers' corners (px) averaged over frames, by id: those seen in most frames; None if fewer
    than two are."""
    import cv2

    from chefbot.onboard.calibration.targets import hand_card_points

    ids_wanted = set(hand_card_points())
    seen: dict[int, list] = {i: [] for i in ids_wanted}
    for _ in range(frames):
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(cam.frame()["bgr"], cv2.COLOR_BGR2GRAY))
        for i, c in zip([] if ids is None else ids.ravel(), corners):
            if int(i) in seen:
                seen[int(i)].append(c[0])
    out = {i: np.mean(v, 0) for i, v in seen.items() if len(v) >= frames // 2}
    return out if len(out) >= 2 else None


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


def record(out: Path, dictionary: str, marker_id: int, size: float | None, card: bool = False) -> None:
    from chefbot.onboard.calibration.arm import Arm, Stop, in_keep_out
    from chefbot.onboard.calibration.camera import POLE_D455, Camera
    from chefbot.onboard.calibration.wrist import corners_now

    arm, det = Arm(), _detector(dictionary)
    marker = {"target": "hand_card", "dictionary": dictionary} if card else {"dictionary": dictionary, "id": marker_id, "size_m": size}
    seen = (lambda frames=FRAMES: card_corners(cam, det, frames)) if card else (lambda frames=FRAMES: corners_now(cam, det, marker_id, frames))
    samples = []
    with Camera(POLE_D455) as cam:
        q_look = arm.q()
        if in_keep_out(q_look, MARKER_MARGIN):
            raise SystemExit("the hand (with the marker) is too near the pole: start further from it")
        if seen() is None:
            raise SystemExit("the pole camera doesn't see the " + ("hand card (both markers)" if card else f"{dictionary} id {marker_id}") + ": run look, turn it to the camera")
        poses = [q_look] + poses_around(q_look)
        print(f"{len(poses)} poses")
        save = lambda: out.write_text(json.dumps({"K": cam.K.tolist(), "D": cam.D.tolist(), "marker": marker, "samples": samples}, indent=1))
        try:
            for i, q in enumerate(poses):
                arm.move_to(q, f"pose {i + 1}/{len(poses)}", confirm=i == 1, margin=MARKER_MARGIN)
                arm.still(0.5)
                c = seen()
                if c is None:
                    print("  marker not seen: skipped")
                    continue
                q_still = arm.still(0.5)["q"]
                samples.append({"q": q_still, "markers": {str(i): v.tolist() for i, v in c.items()}} if card else {"q": q_still, "corners": c.tolist()})
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
    if d.get("marker", {}).get("target") == "hand_card":
        return solve_card(d, K, D, skip)
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


def solve_card(d: dict, K, D, skip=()) -> dict:
    """The camera (Y, in the arm base) and the card on the flange (Z) from every pose's card corners: a Park start
    from each pose's card pose, then least squares on all the corners' reprojection."""
    import cv2

    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.targets import hand_card_points

    pts = hand_card_points()
    samples = [s for i, s in enumerate(d["samples"]) if i + 1 not in set(skip)]
    if len(samples) < 5:
        raise SystemExit(f"{len(samples)} poses: need at least 5")
    G = [fr3.flange(s["q"]) for s in samples]
    obs = [(np.concatenate([pts[int(i)] for i in s["markers"]]), np.concatenate([np.array(c) for c in s["markers"].values()])) for s in samples]
    C = []
    for obj, img in obs:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_SQPNP)
        C.append(_T(cv2.Rodrigues(rvec)[0], tvec))
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    Y = ax_xb([G[j] @ np.linalg.inv(G[i]) for i, j in pairs], [C[j] @ np.linalg.inv(C[i]) for i, j in pairs])
    Zs = [np.linalg.inv(g) @ Y @ c for g, c in zip(G, C)]
    Z = _T(Zs[0][:3, :3], np.mean([z[:3, 3] for z in Zs], 0))

    def residual(x):
        Yx, Zx = _T(cv2.Rodrigues(x[:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12])
        out = []
        for g, (obj, img) in zip(G, obs):
            Cx = _pole_chain(g, Yx, Zx)
            out.append((cv2.projectPoints(obj, cv2.Rodrigues(Cx[:3, :3])[0], Cx[:3, 3], K, D)[0].reshape(-1, 2) - img).ravel())
        return np.concatenate(out)

    x = np.concatenate([cv2.Rodrigues(Y[:3, :3])[0].ravel(), Y[:3, 3], cv2.Rodrigues(Z[:3, :3])[0].ravel(), Z[:3, 3]])
    r, lam = residual(x), 1e-3
    for _ in range(40):  # Levenberg-Marquardt
        J = np.stack([(residual(x + np.eye(12)[k] * 1e-6) - r) / 1e-6 for k in range(12)], 1)
        H = J.T @ J
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H)), J.T @ r)
        r_new = residual(x + step)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam / 3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 5
    Y, Z = _T(cv2.Rodrigues(x[:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12])
    px = [np.sqrt(np.mean(np.sum(rr.reshape(-1, 2) ** 2, 1))) for rr in np.split(r, np.cumsum([2 * len(o[1]) for o in obs])[:-1])]
    T_base = _arm_mount() @ Y
    centers = np.array([(g @ Z)[:3, 3] for g in G])
    out = {"poses": len(samples), "skipped": sorted(skip), "target": "hand_card",
           "T_base_camera": np.round(T_base, 5).tolist(), "T_armbase_camera": np.round(Y, 5).tolist(),
           "T_flange_card": np.round(Z, 5).tolist(),
           "camera_in_base_m": np.round(T_base[:3, 3], 4).tolist(),
           "camera_looks_down_deg": round(float(np.degrees(np.arcsin(-T_base[2, 2]))), 1),
           "reprojection_px": {"rms": round(float(np.sqrt(np.mean(np.square(px)))), 2), "max_pose": round(float(max(px)), 2)},
           "reprojection_rms_px_per_pose": np.round(px, 2).tolist(),
           "card_on_flange_m": np.round(Z[:3, 3], 4).tolist(),
           "card_positions_spread_m": np.round(centers.std(0), 3).tolist()}  # fmt: skip
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
    b.add_argument("--card", action="store_true", help="the hand card (targets.py) instead of one marker")
    c = sub.add_parser("solve")
    c.add_argument("data", type=Path)
    c.add_argument("--skip", type=int, nargs="*", default=[])
    c.add_argument("--size", type=float, default=None)
    args = ap.parse_args()
    if args.cmd == "look":
        look(args.out)
    elif args.cmd == "record":
        record(args.out, args.dict, args.id, args.size, args.card)
    else:
        solve(args.data, args.skip, args.size)


if __name__ == "__main__":
    main()
