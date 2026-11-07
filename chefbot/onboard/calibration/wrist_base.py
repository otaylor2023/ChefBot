"""The wrist camera calibrated against the card glued on the base by the pole (targets.BASE_CARD: ids 1, 2, 55 mm,
70 mm apart): no board to set out, its size known.

    python -m chefbot.onboard.calibration.wrist_base record out.json   # ready pose, then the survey's view of the card
                                                                       # (Enter for each), then ~16 views (Enter once)
    python -m chefbot.onboard.calibration.wrist_base record out.json --from-here   # from where the arm is (put it
        # where the wrist camera sees both markers: launch.sh/arm.sh's f floats it, h holds it), then the views (it solves at the end)
    python -m chefbot.onboard.calibration.wrist_base look       # which of the card's markers the wrist camera sees, live
    python -m chefbot.onboard.calibration.wrist_base solve out.json [--skip 3]

The views: the camera 0.25-0.45 m from the card, up to 35 deg off straight on, turned about its view axis, each one
and every move between them clear of the pole's keep-out and 5 cm above the plate (survey.clear, survey.path_ok),
the wrist turned at most 100 deg from the start (the camera's cable) and no joint over 90 deg in one move.
The solve: the camera on the flange and the card on the base, least squares on all the corners' reprojection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration.wrist import FRAMES, _park, _rot, _T, ik_any, wrist_mount

VIEWS = 16
TILTS_DEG = (0, 15, 25, 35)
DISTANCES = (0.25, 0.35, 0.45)
ROLLS_DEG = (0, 60, 120, 180, 240, 300)
MAX_ROLL = np.radians(100)  # the wrist (joint 7) within this of where it starts: the camera's cable
MAX_MOVE = np.radians(90)  # no joint further than this in one move


def _detector():
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def card_corners(cam, det, frames: int = FRAMES) -> dict[int, np.ndarray] | None:
    """The base card's markers' corners averaged over frames, by id: those seen in most frames; None if not both."""
    import cv2

    from chefbot.onboard.calibration.targets import base_card_points

    seen: dict[int, list] = {i: [] for i in base_card_points()}
    for _ in range(frames):
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(cam.frame()["bgr"], cv2.COLOR_BGR2GRAY))
        for i, c in zip([] if ids is None else ids.ravel(), corners):
            if int(i) in seen:
                seen[int(i)].append(c[0])
    out = {i: np.mean(v, 0) for i, v in seen.items() if len(v) >= frames // 2}
    return out if len(out) == len(seen) else None


def _card_pose(corners: dict, K, D) -> np.ndarray:
    """The card in the camera, from its corners in one image."""
    import cv2

    from chefbot.onboard.calibration.targets import base_card_points

    pts = base_card_points()
    obj = np.concatenate([pts[i] for i in corners])
    img = np.concatenate([np.asarray(c, float) for c in corners.values()])
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), np.asarray(D, float), flags=cv2.SOLVEPNP_SQPNP)
    return _T(cv2.Rodrigues(rvec)[0], tvec)


def plan_views(card: np.ndarray, q_start, n: int = VIEWS) -> list[np.ndarray]:
    """Joint poses viewing the card (its pose in the arm base) from spread distances, tilts and rolls, each reached
    clear from the one before (straight on at each distance first, then tilted from around)."""
    from chefbot.onboard.calibration.survey import _view, clear, path_ok

    X_inv = np.linalg.inv(wrist_mount())
    center, normal = card[:3, 3], card[:3, 2]
    wanted = [(0, 0, d, r) for d in DISTANCES for r in (0, 120, 240)]
    wanted += [(t, az, 0.35, ROLLS_DEG[k % len(ROLLS_DEG)]) for t in TILTS_DEG[1:] for k, az in enumerate(range(0, 360, 60))]
    q_start = np.asarray(q_start, float)
    poses, prev = [], q_start
    for tilt, az, dist, roll in wanted:
        if len([q for q in poses if q is not q_start]) >= n:
            break
        for r in [roll] + [x for x in ROLLS_DEG if x != roll]:  # its own roll first, then any
            T = _view(center, normal, tilt, az, dist) @ _T(_rot([0, 0, 1], r), [0, 0, 0]) @ X_inv
            ok = lambda q, p=prev: (abs(q[6] - q_start[6]) < MAX_ROLL and clear(q)
                                    and ((np.abs(q - p).max() < MAX_MOVE and path_ok(p, q)) or (np.abs(q - q_start).max() < MAX_MOVE and path_ok(q_start, q))))  # fmt: skip
            q = ik_any(T, prev, ok)
            if q is None:
                continue
            if not (np.abs(q - prev).max() < MAX_MOVE and path_ok(prev, q)):  # by way of the start
                poses.append(q_start)
            poses.append(q)
            prev = q
            break
    return poses


def record(out: Path, from_here: bool = False) -> None:
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import Arm, Stop
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera
    from chefbot.onboard.calibration.survey import POSES_FILE, READY, path_ok

    arm, det = Arm(), _detector()
    if from_here:  # where the operator put it, seeing the card
        q_view = arm.q()
        print(f"from here: {np.round(q_view, 3).tolist()}")
    else:
        q_view = np.array(json.loads(POSES_FILE.read_text())["pole"])  # the survey's view by the pole: it saw the card
        arm.move_to(READY, "the ready pose")
        if not path_ok(READY, q_view):
            raise SystemExit("the survey's view of the card isn't clear from the ready pose any more")
        arm.move_to(q_view, "the survey's view of the card by the pole")
    samples = []
    with Camera(WRIST_D405) as cam:
        c = card_corners(cam, det)
        if c is None:
            raise SystemExit("the wrist camera doesn't see both of the base card's markers (ids 1, 2) from here"
                             + ("" if from_here else ": put the arm where it does and use --from-here"))
        card = fr3.flange(arm.q()) @ wrist_mount() @ _card_pose(c, cam.K, cam.D)  # roughly: through the old calibration
        print("planning the views (up to a minute)...", flush=True)
        poses = plan_views(card, arm.q())
        print(f"the card at {np.round(card[:3, 3], 3).tolist()} m; {len(poses)} views")
        save = lambda: out.write_text(json.dumps({"K": cam.K.tolist(), "D": cam.D.tolist(), "target": "base_card", "samples": samples}, indent=1))
        visited = [arm.q()]
        try:
            for i, q in enumerate([arm.q()] + poses):
                if i:
                    arm.move_to(q, f"view {i}/{len(poses)}", confirm=i == 1)
                    visited.append(q)
                arm.still(0.3)
                c = card_corners(cam, det)
                if c is None:
                    print("  the card not seen: skipped")
                    continue
                samples.append({"q": arm.still(0.3)["q"], "markers": {str(k): v.tolist() for k, v in c.items()}})
                print(f"  recorded ({len(samples)} so far)")
                save()
        except (Stop, KeyboardInterrupt):
            arm.hold()
            print("stopped: holding")
        finally:
            save()
            print(f"{len(samples)} views in {out}")
        # back to the survey's view: straight there if that's clear, else the way it came (each step checked going)
        from chefbot.onboard.calibration.survey import path_ok as ok

        back = [q_view] if ok(arm.q(), q_view) else list(reversed(visited[:-1]))
        for q in back:
            arm.move_to(q, "back toward the survey's view", confirm=False)
        if not from_here:
            arm.move_to(READY, "back to the ready pose", confirm=False)
    if len(samples) >= 4:  # the fit, at once (solve again with --skip to drop views)
        print("\nsolving:", flush=True)
        solve(out)
    else:
        print(f"only {len(samples)} views: not enough to solve")


def look() -> None:
    """Twice a second until Ctrl-C: which of the base card's markers the wrist camera sees, and how big (no motion)."""
    import time

    import cv2

    from chefbot.onboard.calibration.camera import WRIST_D405, Camera
    from chefbot.onboard.calibration.targets import base_card_points

    det, wanted = _detector(), sorted(base_card_points())
    with Camera(WRIST_D405) as cam:
        try:
            while True:
                corners, ids, _ = det.detectMarkers(cv2.cvtColor(cam.frame()["bgr"], cv2.COLOR_BGR2GRAY))
                got = {int(i): c[0] for i, c in zip([] if ids is None else ids.ravel(), corners) if int(i) in wanted}
                sizes = {i: f"{np.linalg.norm(c[0] - c[1]):.0f} px" for i, c in got.items()}
                print(("BOTH  " if len(got) == len(wanted) else "      ") + f"card markers seen {sorted(got)} of {wanted} {sizes}", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def solve(path: Path, skip=()) -> dict:
    """X (the camera in the flange) and M (the card in the arm base): Park from each view's card pose, then
    Levenberg-Marquardt on every corner's reprojection."""
    import cv2

    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.targets import base_card_points

    d = json.loads(path.read_text())
    K, D = np.array(d["K"]), np.array(d["D"])
    pts = base_card_points()
    samples = [s for i, s in enumerate(d["samples"]) if i + 1 not in set(skip)]
    if len(samples) < 6:
        raise SystemExit(f"{len(samples)} views: need at least 6")
    G = [fr3.flange(s["q"]) for s in samples]
    obs = [(np.concatenate([pts[int(i)] for i in s["markers"]]), np.concatenate([np.array(c) for c in s["markers"].values()])) for s in samples]
    C = [_card_pose({int(k): v for k, v in s["markers"].items()}, K, D) for s in samples]
    X = _park(G, C)
    Ms = [g @ X @ c for g, c in zip(G, C)]
    M = _T(Ms[0][:3, :3], np.mean([m[:3, 3] for m in Ms], 0))

    def residual(x):
        Xx, Mx = _T(cv2.Rodrigues(x[:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12])
        out = []
        for g, (obj, img) in zip(G, obs):
            Cx = np.linalg.inv(g @ Xx) @ Mx
            out.append((cv2.projectPoints(obj, cv2.Rodrigues(Cx[:3, :3])[0], Cx[:3, 3], K, D)[0].reshape(-1, 2) - img).ravel())
        return np.concatenate(out)

    x = np.concatenate([cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], cv2.Rodrigues(M[:3, :3])[0].ravel(), M[:3, 3]])
    r, lam = residual(x), 1e-3
    for _ in range(50):  # Levenberg-Marquardt
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
    X, M = _T(cv2.Rodrigues(x[:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12])
    per_view = [np.sqrt(np.mean(np.sum(rr.reshape(-1, 2) ** 2, 1))) for rr in np.split(r, np.cumsum([2 * len(o[1]) for o in obs])[:-1])]
    out = {"views": len(samples), "skipped": sorted(skip), "target": "base_card", "T_flange_camera": np.round(X, 5).tolist(),
           "camera_from_flange_m": np.round(X[:3, 3], 4).tolist(),
           "turn_about_flange_z_deg": round(float(np.degrees(np.arctan2(X[1, 0], X[0, 0]))), 1),
           "camera_axis_from_flange_z_deg": round(float(np.degrees(np.arccos(np.clip(X[2, 2], -1, 1)))), 1),
           "reprojection_px": {"rms": round(float(np.sqrt(np.mean(np.square(per_view)))), 2), "max_view": round(float(max(per_view)), 2)},
           "reprojection_rms_px_per_view": np.round(per_view, 2).tolist(),
           "card_in_arm_base_m": np.round(M[:3, 3], 4).tolist(), "T_armbase_card": np.round(M, 5).tolist()}  # fmt: skip
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("out", type=Path)
    r.add_argument("--from-here", action="store_true", help="start where the arm is (it sees both markers), not the survey's view")
    sub.add_parser("look")
    s = sub.add_parser("solve")
    s.add_argument("data", type=Path)
    s.add_argument("--skip", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if args.cmd == "look":
        look()
    elif args.cmd == "record":
        record(args.out, args.from_here)
    else:
        solve(args.data, args.skip)


if __name__ == "__main__":
    main()
