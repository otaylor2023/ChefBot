"""A camera's mount kept current from a marker card on the hand: the arm's joints say where the card's corners are on
the base (the card's pose on the flange calibrated once, pole.py), the camera's frames where they are in its image,
and the camera's pose is what fits both.

    tracker = MountTracker(T_flange_card, card_points, pose=from_config)
    tracker.update(rgb, K, arm_q, arm_qd, flange_in_base)   # each frame from the camera: True if it updated the pose
    tracker.pose                                            # the camera in the base

Not from one frame: a small card's own orientation in the image is poor (13 deg off, for 34 px markers at 0.6 m),
and the camera's pose from it worse. The tracker pools the corners of the recent frames (the arm still in each) and
fits the camera to all of them: its full pose once the hand has been in places spread over the view (3 cm both ways:
on the real arm, 2026-11-03, 18 frames spread 5.6 x 4.1 cm brought a pose knocked 2 cm, 1 deg to 3 mm, 0.5 deg of
the calibration; the 5 cm this started with never let it), until then
only its rotation (its position kept), which the image pins well and which is how a clamped camera mostly drifts. A
run of frames that no longer fit (the camera knocked) clears the pool and starts over from them, with a warning.
"""

from __future__ import annotations

import collections
import logging

import numpy as np

log = logging.getLogger(__name__)


class MountTracker:
    def __init__(self, T_flange_card, card_points: dict[int, np.ndarray], dictionary: str = "DICT_4X4_50", pose=None,
                 min_markers: int = 2, still_qd: float = 0.02, window: int = 40, spread_m: float = 0.03,
                 moved_px: float = 4.0, moved_after: int = 3):  # fmt: skip
        import cv2

        self.T_flange_card = np.asarray(T_flange_card, float)
        self.points = {int(k): np.asarray(v, float) for k, v in card_points.items()}
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
        self.min_markers, self.still_qd, self.spread_m = min_markers, still_qd, spread_m
        self.moved_px, self.moved_after = moved_px, moved_after
        self.frames: collections.deque = collections.deque(maxlen=window)  # (base points Nx3, image points Nx2, K, D)
        self.misfits: list = []
        self.pose = None if pose is None else np.asarray(pose, float)  # the camera in the base: the config's to start
        self.updates, self.moved, self.full = 0, False, False

    def observe(self, rgb, K, arm_q, flange_in_base, D=None):
        """The card's corners on the base (from the joints) and in this image; None if too few markers are seen."""
        import cv2

        gray = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None
        found = [(int(i), c[0]) for i, c in zip(ids.ravel(), corners) if int(i) in self.points]
        if len(found) < self.min_markers:
            return None
        T = np.asarray(flange_in_base(np.asarray(arm_q, float))) @ self.T_flange_card
        base = np.concatenate([self.points[i] @ T[:3, :3].T + T[:3, 3] for i, _ in found])
        image = np.concatenate([c for _, c in found]).astype(np.float64)
        return base, image, np.asarray(K, float), np.zeros(5) if D is None else np.asarray(D, float)

    def update(self, rgb, K, arm_q, arm_qd, flange_in_base, D=None) -> bool:
        if arm_q is None or (arm_qd is not None and np.abs(arm_qd).max() > self.still_qd):
            return False  # the arm moving: its joints and the image may not be the same moment
        ob = self.observe(rgb, K, arm_q, flange_in_base, D)
        if ob is None:
            return False
        if self.pose is not None and self.frames and _rms_px(self.pose, *ob) > self.moved_px:
            self.misfits.append(ob)  # doesn't fit: one bad frame, or the camera moved
            if len(self.misfits) < self.moved_after:
                return False
            log.warning("camera mount moved (%d frames off by over %.0f px): re-estimating", len(self.misfits), self.moved_px)
            self.frames.clear()
            self.frames.extend(self.misfits)
            self.moved = True
        else:
            self.frames.append(ob)
        self.misfits = []
        self.pose, self.full = self._fit()
        self.updates += 1
        return True

    def _fit(self) -> tuple[np.ndarray, bool]:
        """The camera's pose from the pooled frames: in full once the card's positions spread, else its rotation."""
        centers = np.array([f[0].mean(0) for f in self.frames])
        spread = np.sort(np.linalg.svd(centers - centers.mean(0), compute_uv=False))[::-1] / np.sqrt(len(centers))
        full = len(spread) >= 2 and spread[1] > self.spread_m  # spread out in at least two directions
        pose = self.pose if self.pose is not None else _seed(self.frames[0])
        return (_fit_full if full else _fit_rotation)(pose, list(self.frames)), full


def _seed(frame) -> np.ndarray:
    """A first guess from one frame (no config pose to start from)."""
    import cv2

    base, image, K, D = frame
    ok, rvec, tvec = cv2.solvePnP(base, image, K, D, flags=cv2.SOLVEPNP_SQPNP)
    return _pose(rvec, tvec)


def _pose(rvec, tvec) -> np.ndarray:
    """The camera in the base from solvePnP's base-to-camera rvec, tvec."""
    import cv2

    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    P = np.eye(4)
    P[:3, :3], P[:3, 3] = R.T, -R.T @ np.asarray(tvec, float).ravel()
    return P


def _rvec_tvec(pose):
    import cv2

    R = pose[:3, :3].T
    return cv2.Rodrigues(R)[0], (-R @ pose[:3, 3]).reshape(3, 1)


def _residual(pose, frames) -> np.ndarray:
    import cv2

    rvec, tvec = _rvec_tvec(pose)
    return np.concatenate([(cv2.projectPoints(b, rvec, tvec, K, D)[0].reshape(-1, 2) - im).ravel() for b, im, K, D in frames])


def _rms_px(pose, base, image, K, D) -> float:
    r = _residual(pose, [(base, image, K, D)])
    return float(np.sqrt(np.mean(r**2) * 2))


def _fit_full(pose, frames) -> np.ndarray:
    import cv2

    rvec, tvec = _rvec_tvec(pose)
    K, D = frames[-1][2], frames[-1][3]  # one camera: the same intrinsics every frame
    base = np.concatenate([f[0] for f in frames])
    image = np.concatenate([f[1] for f in frames])
    ok, rvec, tvec = cv2.solvePnP(base, image, K, D, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return _pose(rvec, tvec)


def _fit_rotation(pose, frames, iters: int = 10) -> np.ndarray:
    """The camera turned about its own position to fit the frames (Gauss-Newton on a small rotation)."""
    import cv2

    P = pose.copy()
    for _ in range(iters):
        r = _residual(P, frames)
        J = np.empty((r.size, 3))
        for k in range(3):
            w = np.zeros(3)
            w[k] = 1e-6
            Q = P.copy()
            Q[:3, :3] = cv2.Rodrigues(w)[0] @ P[:3, :3]
            J[:, k] = (_residual(Q, frames) - r) / 1e-6
        step = -np.linalg.lstsq(J, r, rcond=None)[0]
        P[:3, :3] = cv2.Rodrigues(step)[0] @ P[:3, :3]
        if np.linalg.norm(step) < 1e-7:
            break
    return P
