"""The marker cards on the robot: their markers' corners in each card's frame (its origin the card's center, x along
its row of markers, y up the printed page, z out of its face; corners in OpenCV's order). Shared by the calibration
(onboard/calibration), the printable targets (onboard/calibration/targets.py) and the mount tracker (mount_track.py)."""

from __future__ import annotations

import numpy as np

DICTIONARY = "DICT_4X4_50"
HAND = {"ids": (4, 5, 6), "marker_m": 0.038, "pitch_m": 0.046, "margin_m": 0.0035}
BASE_CARD = {"ids": (1, 2), "marker_m": 0.055, "pitch_m": 0.070}  # the first hand card, now on the base by the pole


def hand_card_points() -> dict[int, np.ndarray]:
    """The hand card's marker corners in the card's frame (m): its origin the card's center, x along the row of
    markers (id 1 to id 2), y up the printed page, z out of its face; each marker's corners in OpenCV's order (top left,
    top right, bottom right, bottom left, as detectMarkers returns them)."""
    h, pitch = HAND["marker_m"] / 2, HAND["pitch_m"]
    out = {}
    for k, mid in enumerate(HAND["ids"]):
        cx = (k - (len(HAND["ids"]) - 1) / 2) * pitch
        out[mid] = np.array([[cx - h, h, 0], [cx + h, h, 0], [cx + h, -h, 0], [cx - h, -h, 0]], dtype=np.float64)
    return out


def base_card_points() -> dict[int, np.ndarray]:
    """The card on the base (the first, too-big hand card): its markers' corners in its frame, as hand_card_points."""
    h, pitch = BASE_CARD["marker_m"] / 2, BASE_CARD["pitch_m"]
    n = len(BASE_CARD["ids"])
    return {mid: np.array([[cx - h, h, 0], [cx + h, h, 0], [cx + h, -h, 0], [cx - h, -h, 0]], dtype=np.float64)
            for mid, cx in ((mid, (k - (n - 1) / 2) * pitch) for k, mid in enumerate(BASE_CARD["ids"]))}  # fmt: skip
