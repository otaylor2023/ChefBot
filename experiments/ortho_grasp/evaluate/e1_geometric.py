"""E1: offline geometric checks on the full ground-truth cup (spec §14).

Material: inside any convex piece of the cup's collision mesh (the cup is
hollow). Along the segment A-B the material comes in runs. A rim grasp has one
run (a wall between the fingers); a body grasp has two (both walls) with both
fingers outside the cup's outer hull; either counts as straddling, and the
thickness to clear is the span from the first run's start to the last run's end.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

STEP = 0.001  # m along A-B
TIP_CLEAR = 0.003  # m
FINGER_MARGIN = 0.005  # m each side
HEIGHT_MARGIN = 0.01


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of True."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def evaluate(A: np.ndarray, B: np.ndarray, width: float, geometry) -> dict:
    """geometry: observe.CupGeometry (full cloud, hulls, counter, rim)."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    length = float(np.linalg.norm(B - A))
    n = max(2, int(np.ceil(length / STEP)) + 1)
    samples = A + np.linspace(0.0, 1.0, n)[:, None] * (B - A)
    inside = geometry.inside(samples)
    rs = runs(inside)
    outer = ConvexHull(geometry.full).equations
    in_outer = lambda p: bool(np.all(p @ outer[:, :3].T + outer[:, 3] <= 0.0))  # noqa: E731
    tips_out = not inside[0] and not inside[-1]
    if len(rs) == 1:
        straddle = tips_out
    elif len(rs) == 2:
        straddle = tips_out and not in_outer(A) and not in_outer(B)
    else:
        straddle = False
    thickness = (rs[-1][1] - rs[0][0]) * length / (n - 1) if rs else 0.0
    tree = cKDTree(geometry.full)
    dA, dB = tree.query(A)[0], tree.query(B)[0]
    clear_tips = tips_out and dA > TIP_CLEAR and dB > TIP_CLEAR
    width_ok = thickness + 2 * FINGER_MARGIN <= width <= 0.085 + 1e-9
    M = (A + B) / 2
    height_ok = geometry.counter_z + HEIGHT_MARGIN <= M[2] <= geometry.rim_z + HEIGHT_MARGIN
    return {
        "straddle": bool(straddle), "clear_tips": bool(clear_tips), "width_ok": bool(width_ok), "height_ok": bool(height_ok),
        "e1_valid": bool(straddle and clear_tips and width_ok and height_ok),
        "runs": len(rs), "thickness": float(thickness), "tip_clearance_min": float(min(dA, dB)),
        "m_to_surface": float(tree.query(M)[0]),
    }  # fmt: skip
