"""Both cameras at once: the wrist D405 and the pole D455, from one card lying on the counter (the base card, ids 1, 2,
taken off the base) and the hand card on the flange (ids 4-6), all in one fit.

    python -m chefbot.onboard.calibration.joint record out.json      # from where the arm is (it sees the card): ~16
                                                                     # views around it (Enter once), then it solves
    python -m chefbot.onboard.calibration.joint solve out.json [--skip 3 7]

Each view records what each camera sees of the two cards: the wrist camera the counter card; the pole camera the
counter card and, when the hand turns its card toward it, the hand card. The fit solves the wrist camera on the flange
(X), the pole camera in the arm base (P), the hand card on the flange (Z) and the counter card in the arm base (M)
together, on every corner's reprojection in both cameras: the two calibrations then agree on where things on the
counter are, through the same arm. (2026-11-06: calibrated apart, from views behind the arm, they disagreed by 2.5 cm
there, and the hand knocked a cap over.) Start from the ready position (the skills look from it) with the card where
the skills' objects lie, both cameras seeing it: launch.sh/arm.sh's r, then `wrist_base look`.

The output: the wrist camera's T in the flange and the pole camera's in ChefBot's base frame (as
environments/real/config.yaml takes them, OpenCV axes), the hand card on the flange (cameras.pole.track
T_flange_card), the counter card in the arm base; the reprojection per camera; each against the config's now.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chefbot.onboard.calibration.wrist import FRAMES, _rot, _T, wrist_mount


def corners(cam, det, points: dict, frames: int = FRAMES, need: int = 2) -> dict | None:
    """A card's markers' corners (px) averaged over frames, by id: those seen in most frames; None if fewer than
    `need`."""
    import cv2

    seen: dict[int, list] = {i: [] for i in points}
    for _ in range(frames):
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(cam.frame()["bgr"], cv2.COLOR_BGR2GRAY))
        for i, c in zip([] if ids is None else ids.ravel(), cs):
            if int(i) in seen:
                seen[int(i)].append(c[0])
    out = {i: np.mean(v, 0) for i, v in seen.items() if len(v) >= frames // 2}
    return out if len(out) >= need else None


def _pnp(c: dict, points: dict, K, D) -> np.ndarray:
    import cv2

    obj = np.concatenate([points[i] for i in c])
    img = np.concatenate([np.asarray(v, float) for v in c.values()])
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), np.asarray(D, float), flags=cv2.SOLVEPNP_SQPNP)
    return _T(cv2.Rodrigues(rvec)[0], tvec)


def config_poses() -> dict:
    """The config's calibrations now: the wrist T (flange), the pole T (ChefBot's base) and the hand card (flange)."""
    import yaml

    cams = yaml.safe_load((Path(__file__).parents[2] / "environments" / "real" / "config.yaml").read_text())["cameras"]
    return {"wrist": np.array(cams["wrist"]["T"], float), "pole": np.array(cams["pole"]["T"], float),
            "card": np.array(cams["pole"]["track"]["T_flange_card"], float)}  # fmt: skip


def record(out: Path) -> None:
    import cv2

    from chefbot.interface.markers import base_card_points, hand_card_points
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import Arm, Stop
    from chefbot.onboard.calibration.camera import POLE_D455, WRIST_D405, Camera
    from chefbot.onboard.calibration.wrist_base import _detector, plan_views

    arm, det = Arm(), _detector()
    base_pts, hand_pts = base_card_points(), hand_card_points()
    q0 = arm.q()
    samples = []
    with Camera(WRIST_D405) as wcam, Camera(POLE_D455) as pcam:
        c = corners(wcam, det, base_pts, need=len(base_pts))
        if c is None:
            raise SystemExit("the wrist camera doesn't see both of the card's markers (ids 1, 2) from here (wrist_base look)")
        if corners(pcam, det, base_pts, need=len(base_pts)) is None:
            print("note: the pole camera doesn't see the card from here; it may from other views")
        card = fr3.flange(q0) @ wrist_mount() @ _pnp(c, base_pts, wcam.K, wcam.D)  # roughly: through the old calibration
        print("planning the views (up to a minute)...", flush=True)
        poses = plan_views(card, q0)
        print(f"the card at {np.round(card[:3, 3], 3).tolist()} m (arm base); {len(poses)} views")
        meta = {"wrist": {"K": wcam.K.tolist(), "D": wcam.D.tolist()}, "pole": {"K": pcam.K.tolist(), "D": pcam.D.tolist()}}
        save = lambda: out.write_text(json.dumps(meta | {"samples": samples}, indent=1))  # noqa: E731
        visited = [q0]
        try:
            for i, q in enumerate([q0] + poses):
                if i:
                    arm.move_to(q, f"view {i}/{len(poses)}", confirm=i == 1)
                    visited.append(q)
                arm.still(0.3)
                s = {"q": arm.still(0.3)["q"]}
                for key, cam, pts, need in (("wrist_card", wcam, base_pts, 2), ("pole_card", pcam, base_pts, 2),
                                            ("pole_hand", pcam, hand_pts, 2)):  # fmt: skip
                    got = corners(cam, det, pts, need=need)
                    if got is not None:
                        s[key] = {str(k): v.tolist() for k, v in got.items()}
                seen = [k for k in ("wrist_card", "pole_card", "pole_hand") if k in s]
                if seen:
                    samples.append(s)
                print(f"  {', '.join(seen) or 'nothing seen: skipped'} ({len(samples)} views so far)")
                save()
        except (Stop, KeyboardInterrupt):
            arm.hold()
            print("stopped: holding")
        finally:
            save()
            print(f"{len(samples)} views in {out}")
        from chefbot.onboard.calibration.survey import path_ok

        back = [q0] if path_ok(arm.q(), q0) else list(reversed(visited[:-1]))
        for q in back:
            arm.move_to(q, "back to the start", confirm=False)
    if len(samples) >= 6:
        print("\nsolving:", flush=True)
        solve(out)
    else:
        print(f"only {len(samples)} views: not enough to solve")


def _vec(T) -> np.ndarray:
    import cv2

    return np.r_[cv2.Rodrigues(np.asarray(T[:3, :3], float))[0].ravel(), T[:3, 3]]


def _mat(v) -> np.ndarray:
    import cv2

    return _T(cv2.Rodrigues(np.asarray(v[:3], float))[0], v[3:6])


def _apart(A, B) -> tuple[float, float]:
    """(mm, deg) between two poses."""
    import cv2

    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000), float(np.degrees(np.linalg.norm(cv2.Rodrigues(A[:3, :3] @ B[:3, :3].T)[0])))


def fit(samples, Kw, Dw, Kp, Dp, flange, start: dict, iters: int = 60) -> dict:
    """X (wrist camera in the flange), P (pole camera in the arm base), Z (hand card in the flange), M (counter card in
    the arm base), from `start`'s, by Levenberg-Marquardt on the reprojection of every corner in both cameras.
    `flange(q)` is the flange in the arm base. Returns the four and the residuals per camera."""
    import cv2

    from chefbot.interface.markers import base_card_points, hand_card_points

    base_pts, hand_pts = base_card_points(), hand_card_points()
    F = [flange(np.asarray(s["q"], float)) for s in samples]

    def project(T_cam_obj, pts, K, D):
        rvec = cv2.Rodrigues(T_cam_obj[:3, :3])[0]
        return cv2.projectPoints(pts, rvec, T_cam_obj[:3, 3], K, D)[0].reshape(-1, 2)

    def parts(x):
        X, P, Z, M = (_mat(x[6 * k: 6 * k + 6]) for k in range(4))
        out = {"wrist": [], "pole": []}
        for s, Fi in zip(samples, F):
            for key, cam, T, pts, K, D in (("wrist_card", "wrist", np.linalg.inv(Fi @ X) @ M, base_pts, Kw, Dw),
                                            ("pole_card", "pole", np.linalg.inv(P) @ M, base_pts, Kp, Dp),
                                            ("pole_hand", "pole", np.linalg.inv(P) @ Fi @ Z, hand_pts, Kp, Dp)):  # fmt: skip
                if key in s:
                    ids = [int(k) for k in s[key]]
                    obj = np.concatenate([pts[i] for i in ids])
                    img = np.concatenate([np.asarray(s[key][str(i)], float) for i in ids])
                    out[cam].append((project(T, obj, K, D) - img).ravel())
        return out

    def residual(x):
        p = parts(x)
        return np.concatenate([np.concatenate(v) if v else np.zeros(0) for v in p.values()])

    x = np.concatenate([_vec(start[k]) for k in ("X", "P", "Z", "M")])
    r, lam = residual(x), 1e-3
    for _ in range(iters):  # Levenberg-Marquardt (a small ridge: a part no camera sees, e.g. Z, stays where it starts)
        J = np.stack([(residual(x + np.eye(24)[k] * 1e-6) - r) / 1e-6 for k in range(24)], 1)
        H = J.T @ J
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-9 * np.eye(24), J.T @ r)
        r_new = residual(x + step)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam / 3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 5
    X, P, Z, M = (_mat(x[6 * k: 6 * k + 6]) for k in range(4))
    rms = {cam: (float(np.sqrt(np.mean(np.square(np.concatenate(v))))) if v else None) for cam, v in parts(x).items()}
    return {"X": X, "P": P, "Z": Z, "M": M, "rms_px": rms}


def solve(path: Path, skip=()) -> dict:
    from chefbot.interface.markers import base_card_points
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.pole import _arm_mount

    d = json.loads(path.read_text())
    samples = [s for i, s in enumerate(d["samples"]) if i + 1 not in set(skip)]
    Kw, Dw, Kp, Dp = (np.array(d[c][k], float) for c in ("wrist", "pole") for k in ("K", "D"))
    now, mount = config_poses(), _arm_mount()
    first = next((s for s in samples if "wrist_card" in s), None)
    if first is None:
        raise SystemExit("no view has the wrist camera seeing the card")
    c = {int(k): np.asarray(v, float) for k, v in first["wrist_card"].items()}
    M0 = fr3.flange(np.asarray(first["q"], float)) @ now["wrist"] @ _pnp(c, base_card_points(), Kw, Dw)
    start = {"X": now["wrist"], "P": np.linalg.inv(mount) @ now["pole"], "Z": now["card"], "M": M0}
    res = fit(samples, Kw, Dw, Kp, Dp, fr3.flange, start)
    counts = {k: sum(k in s for s in samples) for k in ("wrist_card", "pole_card", "pole_hand")}
    T_pole = mount @ res["P"]
    out = {
        "views": len(samples), "skipped": sorted(skip), "seen": counts,
        "reprojection_rms_px": {k: (round(v, 2) if v is not None else None) for k, v in res["rms_px"].items()},
        "wrist_T_flange_camera": np.round(res["X"], 5).tolist(),
        "pole_T_base_camera": np.round(T_pole, 5).tolist(),
        "T_flange_card": np.round(res["Z"], 5).tolist(),
        "counter_card_in_arm_base_m": np.round(res["M"][:3, 3], 4).tolist(),
        "against_the_config_mm_deg": {"wrist": np.round(_apart(res["X"], now["wrist"]), 2).tolist(),
                                       "pole": np.round(_apart(T_pole, now["pole"]), 2).tolist(),
                                       "hand_card": np.round(_apart(res["Z"], now["card"]), 2).tolist()},  # fmt: skip
    }
    if counts["pole_hand"] == 0:
        out["note"] = "the pole camera never saw the hand card: the hand card's pose is the config's, unfitted"
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record").add_argument("out", type=Path)
    s = sub.add_parser("solve")
    s.add_argument("data", type=Path)
    s.add_argument("--skip", type=int, nargs="*", default=[])
    a = ap.parse_args()
    record(a.out) if a.cmd == "record" else solve(a.data, a.skip)


if __name__ == "__main__":
    main()
