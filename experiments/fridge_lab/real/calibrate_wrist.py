"""Calibrate the wrist fisheye for robot_real.yaml: its lens (K, D at 640x360) and where it sits on the flange.

    # 1. the lens: hold a checkerboard in front of the wrist camera at many angles and distances,
    #    filling the corners of the image too; a view is kept each time the board is found (every 1 s)
    python -m experiments.fridge_lab.real.calibrate_wrist intrinsics --views 25

    # 2. the mount (hand-eye): the arm goes to its ready pose; fix the board where the wrist camera sees it
    #    from there (~30 cm away); the arm moves through small turns and shifts around that pose (the base
    #    doesn't move), and the camera's pose on the flange is solved for
    python -m experiments.fridge_lab.real.calibrate_wrist handeye --K-D wrist_intrinsics.json

Each writes a JSON next to its images and prints the lines for robot_real.yaml (cameras.wrist.K, D, cam_in_flange).
Board: --cols x --rows inner corners, --square meters (default 9x6, 25 mm).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from experiments.fridge_lab import fisheye
from experiments.fridge_lab.real.cameras import H, W

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)  # corner refinement


def board_points(cols: int, rows: int, square: float) -> np.ndarray:
    g = np.zeros((rows * cols, 3), np.float64)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return g


def find_corners(rgb: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (3, 3), (-1, -1), CRIT).reshape(-1, 2)


def project(obj: np.ndarray, rvec, tvec, k: np.ndarray) -> np.ndarray:
    """OpenCV's fisheye model (cv2.fisheye.projectPoints): k = fx, fy, cx, cy, k1..k4."""
    P = obj @ Rotation.from_rotvec(rvec).as_matrix().T + tvec
    a, b = P[:, 0] / P[:, 2], P[:, 1] / P[:, 2]
    r = np.hypot(a, b)
    th = np.arctan(r)
    thd = th * (1 + k[4] * th**2 + k[5] * th**4 + k[6] * th**6 + k[7] * th**8)
    s = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
    return np.stack([k[0] * s * a + k[2], k[1] * s * b + k[3]], 1)


def solve_intrinsics(corners: list[np.ndarray], obj: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fisheye calibration at 640x360 by least squares over the lens and each view's board pose: (K, D, rms px).
    (Not cv2.fisheye.calibrate, which fails in OpenCV 5.0.)"""
    from scipy.optimize import least_squares

    K0 = fisheye.K_of(fisheye.nominal_f())
    k0 = np.array([K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], *fisheye.D], dtype=float)
    poses = []
    for c in corners:  # each view's pose under the nominal lens
        T = board_in_camera(c, obj, K0, fisheye.D)
        poses += [*Rotation.from_matrix(T[:3, :3]).as_rotvec(), *T[:3, 3]]
    x0 = np.concatenate([k0, poses])

    def resid(x):
        k = x[:8]
        return np.concatenate([(project(obj, x[8 + 6 * i : 11 + 6 * i], x[11 + 6 * i : 14 + 6 * i], k) - c).ravel()
                               for i, c in enumerate(corners)])  # fmt: skip

    sol = least_squares(resid, x0, method="lm", x_scale="jac")
    k = sol.x[:8]
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]])
    return K, k[4:], float(np.sqrt(np.mean(sol.fun.reshape(-1, 2) ** 2) * 2))


def board_in_camera(corners: np.ndarray, obj: np.ndarray, K, D) -> np.ndarray | None:
    """The board's pose in the camera (OpenCV axes), from fisheye pixels: undistort to normalized rays, then PnP."""
    und = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2).astype(np.float64), np.asarray(K), np.asarray(D, dtype=np.float64))
    ok, rvec, tvec = cv2.solvePnP(obj, und.reshape(-1, 2), np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T


def solve_handeye(flange_in_root: list[np.ndarray], board_in_cam: list[np.ndarray]) -> np.ndarray:
    """The camera's pose X in the flange frame (eye in hand, board fixed in the arm base frame: F_i X B_i is the
    same for every view). Park and Martin's closed form over all view pairs, then least squares on that agreement.
    (Not cv2.calibrateHandEye, which OpenCV 5.0 lacks.)"""
    from scipy.optimize import least_squares

    M, rows, rhs = np.zeros((3, 3)), [], []
    pairs = [(i, j) for i in range(len(board_in_cam)) for j in range(i + 1, len(board_in_cam))]
    for i, j in pairs:  # A X = X B, A = F_j^-1 F_i (flange motion), B = B_j B_i^-1 (camera motion)
        A = np.linalg.inv(flange_in_root[j]) @ flange_in_root[i]
        B = board_in_cam[j] @ np.linalg.inv(board_in_cam[i])
        M += np.outer(Rotation.from_matrix(B[:3, :3]).as_rotvec(), Rotation.from_matrix(A[:3, :3]).as_rotvec())
    w, V = np.linalg.eigh(M.T @ M)
    Rx = V @ np.diag(w ** -0.5) @ V.T @ M.T
    for i, j in pairs:
        A = np.linalg.inv(flange_in_root[j]) @ flange_in_root[i]
        B = board_in_cam[j] @ np.linalg.inv(board_in_cam[i])
        rows.append(A[:3, :3] - np.eye(3))
        rhs.append(Rx @ B[:3, 3] - A[:3, 3])
    tx = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]

    def T_of(v):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6]
        return T

    def resid(v):  # every view's board pose against a common one (m, and rad scaled to ~m at 10 cm)
        X, Tb = T_of(v[:6]), T_of(v[6:])
        out = []
        for F, B in zip(flange_in_root, board_in_cam):
            E = np.linalg.inv(Tb) @ F @ X @ B
            out += [*E[:3, 3], *(0.1 * Rotation.from_matrix(E[:3, :3]).as_rotvec())]
        return np.array(out)

    X0 = T_of([*Rotation.from_matrix(Rx).as_rotvec(), *tx])
    Tb0 = flange_in_root[0] @ X0 @ board_in_cam[0]
    sol = least_squares(resid, [*Rotation.from_matrix(X0[:3, :3]).as_rotvec(), *X0[:3, 3],
                                *Rotation.from_matrix(Tb0[:3, :3]).as_rotvec(), *Tb0[:3, 3]])  # fmt: skip
    return T_of(sol.x[:6])


def handeye_residual(flange_in_root, board_in_cam, X) -> tuple[float, float]:
    """How consistent the board's pose in the arm base frame is across views: (position spread m, angle spread deg)."""
    boards = [F @ X @ B for F, B in zip(flange_in_root, board_in_cam)]
    p = np.array([b[:3, 3] for b in boards])
    r = Rotation.from_matrix([b[:3, :3] for b in boards])
    return float(np.linalg.norm(p - p.mean(0), axis=1).max()), float(np.degrees((r * r.mean().inv()).magnitude().max()))


def _yaml_lines(**kv) -> str:
    return yaml.safe_dump({"cameras": {"wrist": {k: np.round(np.asarray(v), 6).tolist() for k, v in kv.items()}}}, default_flow_style=None)


def intrinsics(args) -> None:
    from experiments.fridge_lab.real.cameras import UsbFisheyeCamera

    cfg = yaml.safe_load(Path(args.config).read_text())["cameras"]["wrist"]
    cam = UsbFisheyeCamera(cfg["serial"], cfg.get("width", 1920), cfg.get("height", 1080))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    obj, views = board_points(args.cols, args.rows, args.square), []
    try:
        print(f"Show the board to the wrist camera; {args.views} views needed.")
        while len(views) < args.views:
            rgb, _, _ = cam.frame()
            c = find_corners(rgb, args.cols, args.rows)
            if c is not None:
                views.append(c)
                cv2.imwrite(str(out / f"view{len(views):02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"view {len(views)}: board at {np.round(c.mean(0)).astype(int).tolist()}")
                time.sleep(1.0)
    finally:
        cam.close()
    K, D, rms = solve_intrinsics(views, obj)
    (out / "wrist_intrinsics.json").write_text(json.dumps({"K": K.tolist(), "D": D.tolist(), "rms_px": rms, "size": [W, H]}, indent=1))
    print(f"rms {rms:.2f} px (aim for < 0.5)\n" + _yaml_lines(K=K, D=D))


def handeye(args) -> None:
    from experiments.fridge_lab.real.host import RealHost, quat_wxyz
    from experiments.fridge_lab.real.run_real import load_config

    cal = json.loads(Path(args.K_D).read_text())
    K, D = np.array(cal["K"]), np.array(cal["D"])
    cfg = load_config(args.config, args)
    cfg["safety"]["speed"] = min(cfg["safety"]["speed"], 0.3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    host = RealHost(cfg, out, record=False)
    obj = board_points(args.cols, args.rows, args.square)
    Fs, Bs = [], []
    try:
        host.reset_robot()
        host.find_start()
        p = host.proprio()
        F0, base = np.array(p["flange_pose_odom"]), p["base_pose_odom"]
        rng = np.random.default_rng(0)
        for i in range(args.poses):
            F = F0.copy()
            if i:  # the first view is the start pose
                F[:3, :3] = Rotation.from_rotvec(np.radians(rng.uniform(-12, 12, 3))).as_matrix() @ F0[:3, :3]
                F[:3, 3] += rng.uniform(-0.04, 0.04, 3)
            row = [*F[:3, 3], *quat_wxyz(F[:3, :3]), 0.0, *base]
            for _ in range(45):  # 3 s to get there and settle
                host.step(row)
            m = host._measure()
            rgb = host.cams["wrist"].frame()[0]
            c = find_corners(rgb, args.cols, args.rows)
            B = None if c is None else board_in_camera(c, obj, K, D)
            print(f"pose {i}: {'board found' if B is not None else 'no board'}")
            if B is not None:
                Fs.append(host.kin.fk(m["q"]))  # the flange in the arm base frame (the base is still)
                Bs.append(B)
                cv2.imwrite(str(out / f"pose{i:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        row = [*F0[:3, 3], *quat_wxyz(F0[:3, :3]), 0.0, *base]
        for _ in range(45):
            host.step(row)
    finally:
        host.close()
    if len(Fs) < 5:
        raise SystemExit(f"only {len(Fs)} views of the board; need at least 5")
    X = solve_handeye(Fs, Bs)
    dp, da = handeye_residual(Fs, Bs, X)
    (out / "wrist_handeye.json").write_text(json.dumps({"cam_in_flange": X.tolist(), "views": len(Fs), "spread_m": dp, "spread_deg": da}, indent=1))
    print(f"{len(Fs)} views; the board stays within {dp * 100:.1f} cm / {da:.1f} deg across them (aim for < 1 cm / 1 deg)\n"
          + _yaml_lines(cam_in_flange=X))  # fmt: skip


def main() -> None:
    from experiments.fridge_lab.real.run_real import CONFIG

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("intrinsics", "handeye"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, default=CONFIG)
        p.add_argument("--out", type=Path, default=Path.home() / "fridge_lab" / "real_calib" / time.strftime("%Y%m%d_%H%M%S"))
        p.add_argument("--cols", type=int, default=9)
        p.add_argument("--rows", type=int, default=6)
        p.add_argument("--square", type=float, default=0.025)
        if name == "intrinsics":
            p.add_argument("--views", type=int, default=25)
        else:
            p.add_argument("--K-D", dest="K_D", required=True, help="wrist_intrinsics.json from the intrinsics step")
            p.add_argument("--poses", type=int, default=15)
            p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    {"intrinsics": intrinsics, "handeye": handeye}[args.cmd](args)


if __name__ == "__main__":
    main()
