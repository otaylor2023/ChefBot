"""The wrist D405's pose on the flange (eye-in-hand), from an ArUco marker lying still in front of the robot.

    python -m chefbot.onboard.calibration.wrist look [--pose ready|forward]  # moves the arm (Enter first), then says
                                                                             # which markers the camera sees (look.png)
    python -m chefbot.onboard.calibration.wrist record out.json [--dict DICT_4X4_50 --id 0 --size 0.127]
        # from where the marker is in view: orbits the camera around it (Enter once), corners per pose
    python -m chefbot.onboard.calibration.wrist solve out.json [--skip 12 13] [--size 0.132]
        # the camera in the flange frame; the marker's size fitted unless measured (--size: the black square, m).
        # The camera's offset along its view axis trades off against the size: measure it if you can.

The marker lies flat and stiff (a flimsy one bows). A measured size (its black square) pins the camera's offset
along its view axis, which otherwise trades off against the fitted size. The camera's pose comes out in ChefBot's
convention (environments/real/config.yaml cameras): OpenCV axes (x right, y down, z forward) in the flange frame.
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


DICTIONARY, MARKER_ID = "DICT_4X4_50", 0  # record's defaults (--dict, --id)
MARKER_SIZE = 0.134  # m, the black square, when not given (2026-11-02's flimsy marker, by depth; solve refits it)
FRAMES = 10  # corner averages per pose
ORBIT_TILTS_DEG = (0, 12, 22)  # the camera's view axis from vertical, around the marker
ORBIT_ROLLS_DEG = (-50, 0, 50)  # turns about the view axis (hand-eye needs rotations about different axes)


def detect_markers(gray) -> list[tuple[str, np.ndarray, list]]:
    """Every predefined ArUco dictionary that finds a marker: (name, ids, corners), most markers first (one marker
    can decode in several dictionaries: the smallest that has it is the one)."""
    import cv2

    found = []
    for name in (n for n in dir(cv2.aruco) if n.startswith("DICT_")):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(gray)
        if ids is not None:
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


def _detect(det, gray, marker_id: int):
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or marker_id not in ids.flatten():
        return None
    return corners[int(np.flatnonzero(ids.flatten() == marker_id)[0])][0]


def corners_now(cam, det, marker_id: int, frames: int = FRAMES):
    """The marker's 4 corners (px), averaged over frames; None if not seen in most of them."""
    import cv2

    seen = []
    for _ in range(frames):
        c = _detect(det, cv2.cvtColor(cam.frame()["bgr"], cv2.COLOR_BGR2GRAY), marker_id)
        if c is not None:
            seen.append(c)
    if len(seen) < frames // 2:
        return None
    return np.mean(seen, 0)


def _rot(axis, deg):
    import cv2

    return cv2.Rodrigues(np.asarray(axis, float) * np.radians(deg))[0]


def ik(q0, T_goal, iters: int = 300) -> np.ndarray | None:
    """Joints putting the flange at T_goal (damped least squares from q0), within the joint limits less a margin."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import LIMITS, MARGIN

    import cv2

    q = np.array(q0, float)
    for _ in range(iters):
        T = fr3.flange(q)
        dp = T_goal[:3, 3] - T[:3, 3]
        dr = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)[0].ravel()
        if np.linalg.norm(dp) < 1e-4 and np.linalg.norm(dr) < 1e-3:
            break
        J = fr3.jacobian(q)
        e = np.concatenate([dp, dr])
        q += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q += 0.05 * (np.mean(LIMITS, 1) - q) * np.array([0, 0, 1, 0, 0, 0, 0])  # keep joint 3 near the middle
    else:
        return None
    lo, hi = LIMITS[:, 0] + MARGIN, LIMITS[:, 1] - MARGIN
    return q if np.all(q > lo) and np.all(q < hi) else None


def orbit(q_look, marker_cam) -> list[np.ndarray]:
    """Joint poses around the look pose: the flange at the look pose's distance from the marker, its z axis tilted
    off the vertical and turned about it, still pointing at the marker (as if the camera looked along the flange's
    z from the flange; near enough to keep the marker in view)."""
    from chefbot.onboard.calibration import fr3
    from chefbot.onboard.calibration.arm import path_clear

    T0 = fr3.flange(q_look)
    target = T0[:3, 3] + T0[:3, :3] @ marker_cam  # the marker in the base, taking the camera as at the flange
    dist = float(np.linalg.norm(target - T0[:3, 3]))
    poses = []
    for tilt in ORBIT_TILTS_DEG:
        for az in ((0,) if tilt == 0 else (0, 90, 180, 270)):
            for roll in ORBIT_ROLLS_DEG:
                # tilt the view axis toward azimuth az (base frame, about the marker), then roll about it
                R = _rot([0, 0, 1], az) @ _rot([0, 1, 0], tilt) @ _rot([0, 0, 1], -az) @ T0[:3, :3] @ _rot([0, 0, 1], roll)
                p = target - R[:, 2] * dist
                if p[0] < 0.15 or p[2] < target[2] + 0.3:  # not over the arm's own base, nor low over the marker
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, p
                q = ik(q_look, T)
                if q is not None and np.abs(q - q_look).max() < np.radians(100) and path_clear(q_look, q):
                    poses.append(q)
    # in order, each move from the one before must be clear too (else go by the look pose)
    out, prev = [], np.asarray(q_look, float)
    for q in poses:
        if not path_clear(prev, q):
            out.append(np.asarray(q_look, float))
        out.append(q)
        prev = q
    return out


def record(out: Path, dictionary: str = DICTIONARY, marker_id: int = MARKER_ID, size: float | None = None) -> None:
    """size: the marker's black square if known (m): saved with the data, and solve then holds it."""
    import json

    import cv2

    from chefbot.onboard.calibration.arm import Arm, Stop
    from chefbot.onboard.calibration.camera import WRIST_D405, Camera

    arm = Arm()
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), cv2.aruco.DetectorParameters())
    marker = {"dictionary": dictionary, "id": marker_id, "size_m": size}
    samples = []
    with Camera(WRIST_D405) as cam:
        q_look = arm.q()
        c = corners_now(cam, det, marker_id)
        if c is None:
            raise SystemExit(f"marker {dictionary} id {marker_id} isn't in view: run look first (or move it under the camera)")
        ok, rvec, tvec = cv2.solvePnP(_object_points(size or MARKER_SIZE), c, cam.K, cam.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        poses = [q_look] + orbit(q_look, tvec.ravel())
        print(f"marker {np.linalg.norm(tvec):.2f} m away; {len(poses)} poses around it")
        try:
            for i, q in enumerate(poses):
                arm.move_to(q, f"pose {i + 1}/{len(poses)}", confirm=i == 1)
                arm.still(0.5)
                c = corners_now(cam, det, marker_id)
                if c is None:
                    print("  marker not seen: skipped")
                    continue
                qs = arm.still(0.5)["q"]
                samples.append({"q": qs, "corners": c.tolist()})
                print(f"  recorded ({len(samples)} so far)")
                out.write_text(json.dumps({"K": cam.K.tolist(), "D": cam.D.tolist(), "marker": marker, "samples": samples}, indent=1))
        except (Stop, KeyboardInterrupt):
            arm.hold()
            print("stopped: holding")
        finally:
            out.write_text(json.dumps({"K": cam.K.tolist(), "D": cam.D.tolist(), "marker": marker, "samples": samples}, indent=1))
            print(f"{len(samples)} poses in {out}")
        arm.move_to(q_look, "back to the look pose", confirm=False)


def _object_points(size: float) -> np.ndarray:
    h = size / 2  # OpenCV's corner order for IPPE_SQUARE: top left, top right, bottom right, bottom left
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def _pnp(samples, K, D, size) -> list[np.ndarray]:
    """The marker in the camera at each pose (the marker's frame: z out of it, x to the right)."""
    import cv2

    out = []
    for s in samples:
        ok, rvec, tvec = cv2.solvePnP(_object_points(size), np.array(s["corners"]), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        out.append(_T(cv2.Rodrigues(rvec)[0], tvec))
    return out


def _park(G: list[np.ndarray], C: list[np.ndarray]) -> np.ndarray:
    """Eye-in-hand AX = XB (Park & Martin): X the camera in the flange, from the flange poses G (in the base) and the
    marker's poses C (in the camera); the marker G_i X C_i is the same at every pose."""
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    return ax_xb([np.linalg.inv(G[j]) @ G[i] for i, j in pairs], [C[j] @ np.linalg.inv(C[i]) for i, j in pairs])


def ax_xb(A: list[np.ndarray], B: list[np.ndarray]) -> np.ndarray:
    """X with A_k X = X B_k for all k, least squares (Park & Martin): rotation, then translation."""
    import cv2

    alpha = np.array([cv2.Rodrigues(a[:3, :3])[0].ravel() for a in A])
    beta = np.array([cv2.Rodrigues(b[:3, :3])[0].ravel() for b in B])
    M = beta.T @ alpha
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(w**-0.5) @ V.T @ M.T
    lhs = np.vstack([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([R @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    return _T(R, np.linalg.lstsq(lhs, rhs, rcond=None)[0])


def _wrist_chain(g, X, M):
    return np.linalg.inv(g @ X) @ M  # the marker in the camera: X the camera in the flange, M the marker in the base


def _refine(samples, K, D, X, Mk, size, fit_size: bool = True, iters: int = 30, chain=_wrist_chain):
    """X, Mk and (fit_size) the marker's size, least squares on the corners' reprojection; chain(flange pose, X, Mk)
    is the marker in the camera (here X the camera in the flange and Mk the marker in the base)."""
    import cv2

    from chefbot.onboard.calibration import fr3

    G = [fr3.flange(s["q"]) for s in samples]
    obs = np.concatenate([np.array(s["corners"]).ravel() for s in samples])

    def unpack(x):
        return (_T(cv2.Rodrigues(x[0:3])[0], x[3:6]), _T(cv2.Rodrigues(x[6:9])[0], x[9:12]), size * np.exp(x[12]))

    def residual(x):
        Xx, Mx, sz = unpack(x)
        out = []
        for g in G:
            C = chain(g, Xx, Mx)
            pts, _ = cv2.projectPoints(_object_points(sz), cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], K, D)
            out.append(pts.ravel())
        return np.concatenate(out) - obs

    x = np.concatenate([cv2.Rodrigues(X[:3, :3])[0].ravel(), X[:3, 3], cv2.Rodrigues(Mk[:3, :3])[0].ravel(), Mk[:3, 3], [0.0]])
    lam = 1e-3
    r = residual(x)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            dx = np.zeros_like(x)
            dx[k] = 1e-6
            J[:, k] = (residual(x + dx) - r) / 1e-6
        if not fit_size:
            J[:, 12] = 0.0
        H, g = J.T @ J, J.T @ r
        step = -np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(x.size), g)
        r_new = residual(x + step)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam / 3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 5
    Xx, Mx, sz = unpack(x)
    return Xx, Mx, sz, r.reshape(len(samples), 4, 2)


def solve(path: Path, skip=(), size: float | None = None) -> dict:
    """skip: poses (from 1) to leave out; size: the marker's black square measured (m), else fitted."""
    import json

    from chefbot.onboard.calibration import fr3

    d = json.loads(path.read_text())
    K, D = np.array(d["K"]), np.array(d["D"])
    size = size or d.get("marker", {}).get("size_m")  # measured: held; else fitted
    samples = [s for i, s in enumerate(d["samples"]) if i + 1 not in set(skip)]
    if len(samples) < 5:
        raise SystemExit(f"{len(samples)} poses: need at least 5")
    G = [fr3.flange(s["q"]) for s in samples]
    C = _pnp(samples, K, D, size or MARKER_SIZE)
    X0 = _park(G, C)
    Ms = [g @ X0 @ c for g, c in zip(G, C)]
    M0 = _T(Ms[0][:3, :3], np.mean([m[:3, 3] for m in Ms], 0))
    X, Mk, size, res = _refine(samples, K, D, X0, M0, size or MARKER_SIZE, fit_size=size is None)
    # the marker from each pose on its own (PnP at the fitted size, through X): it lies still, so this is the error
    pos = np.array([(g @ X @ c)[:3, 3] for g, c in zip(G, _pnp(samples, K, D, size))])
    px = np.linalg.norm(res, axis=2)
    look = X[:3, 2]  # the camera's view axis in the flange frame
    out = {"poses": len(samples), "skipped": sorted(skip), "marker_size_m": round(float(size), 4), "T_flange_camera": np.round(X, 5).tolist(),
           "camera_from_flange_m": np.round(X[:3, 3], 4).tolist(),
           "camera_axis_from_flange_z_deg": round(float(np.degrees(np.arccos(np.clip(look[2], -1, 1)))), 1),
           "reprojection_px": {"rms": round(float(np.sqrt(np.mean(px**2))), 2), "max": round(float(px.max()), 2)},
           "reprojection_rms_px_per_pose": np.round(np.sqrt(np.mean(px**2, 1)), 2).tolist(),
           "marker_in_base_m": np.round(Mk[:3, 3], 4).tolist(),
           "marker_per_pose_spread_mm": np.round(np.linalg.norm(pos - Mk[:3, 3], axis=1) * 1000, 1).tolist(),
           "park_start_vs_fit_mm": round(float(np.linalg.norm(X0[:3, 3] - X[:3, 3]) * 1000), 1)}  # fmt: skip
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("look")
    a.add_argument("--pose", choices=sorted(LOOK_POSES), default="ready")
    a.add_argument("--out", type=Path, default=Path("look.png"))
    b = sub.add_parser("record")
    b.add_argument("out", type=Path)
    b.add_argument("--dict", default=DICTIONARY, help="the marker's ArUco dictionary (look prints it)")
    b.add_argument("--id", type=int, default=MARKER_ID)
    b.add_argument("--size", type=float, default=None, help="the marker's black square, measured (m)")
    c = sub.add_parser("solve")
    c.add_argument("data", type=Path)
    c.add_argument("--skip", type=int, nargs="*", default=[], help="poses (from 1) to leave out")
    c.add_argument("--size", type=float, default=None, help="the marker's black square, measured (m): else fitted")
    args = ap.parse_args()
    if args.cmd == "look":
        look(args.pose, args.out)
    elif args.cmd == "record":
        record(args.out, args.dict, args.id, args.size)
    else:
        solve(args.data, args.skip, args.size)


if __name__ == "__main__":
    main()
