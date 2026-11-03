"""Printable calibration targets, at exact size: print at 100% ("actual size", no fit-to-page) and check the
100 mm scale bar on each with a ruler before mounting. Mount flat and stiff (glue to cardboard or foam board).

    python -m chefbot.onboard.calibration.targets OUT_DIR     # hand_markers.pdf, base_charuco.pdf (US letter)

hand_markers: two 55 mm DICT_4X4_50 markers, ids 1 and 2, 70 mm apart center to center, for the white face of the
    Franka Hand that the pole camera sees (about 19.5 x 9.5 cm): the card 143 x 73 mm. (id 0 is the loose marker.)
base_charuco: a 5 x 4 ChArUco board, 30 mm squares with 22 mm DICT_4X4_50 markers (ids 10-19), for the robot's
    base where the wrist camera can see it: the board 150 x 120 mm, the card 166 x 136 mm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DPI = 600
PAGE_IN = (8.5, 11.0)
DICTIONARY = "DICT_4X4_50"
HAND = {"ids": (1, 2), "marker_m": 0.055, "pitch_m": 0.070, "margin_m": 0.009}
BASE = {"squares": (5, 4), "square_m": 0.030, "marker_m": 0.022, "first_id": 10, "margin_m": 0.008}


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


def _px(m: float) -> int:
    return int(round(m * 1000 / 25.4 * DPI))


def _page() -> np.ndarray:
    return np.full((int(PAGE_IN[1] * DPI), int(PAGE_IN[0] * DPI)), 255, np.uint8)


def _label(img, text: str, x: int, y: int, scale: float = 1.6) -> None:
    import cv2

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 3, cv2.LINE_AA)


def _scale_bar(img, x: int, y: int) -> None:
    """A 100 mm bar with 10 mm ticks."""
    import cv2

    cv2.line(img, (x, y), (x + _px(0.1), y), 0, 4)
    for k in range(11):
        cv2.line(img, (x + _px(0.01 * k), y - (40 if k % 5 == 0 else 20)), (x + _px(0.01 * k), y), 0, 4)
    _label(img, "100 mm: measure this before mounting", x, y + 70)


def _cut_box(img, x0, y0, w, h) -> None:
    """A thin gray cut line around the card."""
    import cv2

    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 170, 2)


def hand_markers() -> np.ndarray:
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    img = _page()
    m, pitch, margin = _px(HAND["marker_m"]), _px(HAND["pitch_m"]), _px(HAND["margin_m"])
    w, h = pitch + m + 2 * margin, m + 2 * margin
    x0, y0 = (img.shape[1] - w) // 2, _px(0.04)
    _cut_box(img, x0, y0, w, h)
    for k, mid in enumerate(HAND["ids"]):
        img[y0 + margin:y0 + margin + m, x0 + margin + k * pitch:x0 + margin + k * pitch + m] = cv2.aruco.generateImageMarker(d, mid, m)
    _label(img, f"ChefBot hand markers: {DICTIONARY} ids {HAND['ids'][0]} (left), {HAND['ids'][1]} (right), "
                f"{HAND['marker_m'] * 1000:.0f} mm, {HAND['pitch_m'] * 1000:.0f} mm apart", x0 - _px(0.02), y0 + h + _px(0.012))  # fmt: skip
    _label(img, "Cut on the gray line. Glue flat to the white face of the hand the pole camera sees.", x0 - _px(0.02), y0 + h + _px(0.022))
    _scale_bar(img, x0, y0 + h + _px(0.05))
    return img


def base_charuco() -> np.ndarray:
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    nx, ny = BASE["squares"]
    ids = np.arange(BASE["first_id"], BASE["first_id"] + (nx * ny) // 2)
    board = cv2.aruco.CharucoBoard((nx, ny), BASE["square_m"], BASE["marker_m"], d, ids)
    bw, bh = _px(nx * BASE["square_m"]), _px(ny * BASE["square_m"])
    art = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    img = _page()
    margin = _px(BASE["margin_m"])
    x0, y0 = (img.shape[1] - bw) // 2 - margin, _px(0.04)
    _cut_box(img, x0, y0, bw + 2 * margin, bh + 2 * margin)
    img[y0 + margin:y0 + margin + bh, x0 + margin:x0 + margin + bw] = art
    _label(img, f"ChefBot base board: ChArUco {nx} x {ny}, {BASE['square_m'] * 1000:.0f} mm squares, "
                f"{BASE['marker_m'] * 1000:.0f} mm {DICTIONARY} markers, ids {ids[0]}-{ids[-1]}", x0, y0 + bh + 2 * margin + _px(0.012))  # fmt: skip
    _label(img, "Cut on the gray line. Glue flat on the base where the wrist camera sees it, clear of the arm.", x0, y0 + bh + 2 * margin + _px(0.022))
    _scale_bar(img, x0, y0 + bh + 2 * margin + _px(0.05))
    return img


def write(out_dir: Path) -> list[Path]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, img in (("hand_markers", hand_markers()), ("base_charuco", base_charuco())):
        p = out_dir / f"{name}.pdf"
        Image.fromarray(img).save(p, "PDF", resolution=DPI)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    for p in write(ap.parse_args().out_dir):
        print(p)


if __name__ == "__main__":
    main()
