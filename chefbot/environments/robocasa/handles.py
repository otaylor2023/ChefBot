"""Ground-truth handle detection from an object's collision mesh.

Stands in for a part segmentation model (SAM or similar on the RGB-D image):
given an object's points, say which of them are its handle. Where to hold the
handle is decided separately, by chefbot.grasping.handle.
"""

from __future__ import annotations

import numpy as np

HANDLE_MAX_WIDTH = 0.05
HANDLE_MIN_LENGTH = 0.05
_SLICE = 0.01  # slices along the object's long axis


def detect_handle(points: np.ndarray) -> np.ndarray | None:
    """(N,) bool mask of the handle's points, or None if the object has no handle.

    A handle is a run of thin slices at one end of the object's long axis (a pan's handle).
    """
    xy = points[:, :2]
    centered = xy - xy.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    long_axis = vectors[:, -1]
    across = np.array([-long_axis[1], long_axis[0]])
    along = xy @ long_axis
    edges = np.arange(along.min(), along.max() + _SLICE, _SLICE)
    # Per slice: thin, wide, or None when the mesh has no points there (a handle
    # is often a few long triangles, with no vertex for several centimeters).
    kinds: list[bool | None] = []
    for start in edges[:-1]:
        piece = points[(along >= start) & (along < start + _SLICE)]
        kinds.append(None if len(piece) < 2 else bool(np.ptp(piece[:, :2] @ across) <= HANDLE_MAX_WIDTH))
    thin = _fill_gaps(kinds)
    best: tuple[int, int] | None = None
    for end in (0, len(thin) - 1):  # a handle starts at an end of the object
        step, i = (1, 0) if end == 0 else (-1, len(thin) - 1)
        count = 0
        while 0 <= i < len(thin) and thin[i]:
            count, i = count + 1, i + step
        if count * _SLICE >= HANDLE_MIN_LENGTH and (best is None or count > best[1]):
            best = (end, count)
    if best is None or best[1] == len(thin):
        return None  # nothing thin, or thin all the way along: no body to stick out of
    end, count = best
    if end == 0:
        return along < edges[count]
    return along >= edges[len(thin) - count]


def _fill_gaps(kinds: list[bool | None]) -> list[bool]:
    """Slices with no points take their neighbors' kind: thin only when the nearest
    known slices on both sides (or the one side, at an end) are thin."""
    filled = []
    for i, kind in enumerate(kinds):
        if kind is not None:
            filled.append(kind)
            continue
        before = next((k for k in reversed(kinds[:i]) if k is not None), None)
        after = next((k for k in kinds[i + 1 :] if k is not None), None)
        filled.append(before is not False and after is not False and (before or after) is True)
    return filled
