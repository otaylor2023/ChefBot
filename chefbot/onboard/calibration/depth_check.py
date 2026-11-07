"""Does a camera's depth agree with its colour image? From a card of known size in view (the base card, ids 1, 2, or the
hand card, ids 4-6): the card's pose from its corners in the colour image (PnP: sizes only, no depth), and from it
the depth each of its pixels should read, against what the depth sensor reads there. Nothing moves.

    python -m chefbot.onboard.calibration.depth_check [--cameras wrist pole] [--frames 10]

Frames as the skills get them: onboard/cameras.py (the server's), 640 x 480, depth aligned to colour, the colour
intrinsics. Run it with the onboard server stopped (arm.sh, not launch.sh: the server holds the cameras). Prints per
camera and card: the distance, the depth sensor's reading over the PnP one (1.000: they agree) and the difference.
A depth that reads long puts points further along the camera's rays: lower, looking down.
"""

from __future__ import annotations

import argparse

import numpy as np


def check(name: str, serial: str, frames: int = 10) -> list[dict]:
    import cv2

    from chefbot.interface.markers import base_card_points, hand_card_points
    from chefbot.onboard.cameras import Camera
    from chefbot.onboard.calibration.wrist_base import _detector

    det = _detector()
    cards = {"base card": base_card_points(), "hand card": hand_card_points()}
    cam = Camera(name, serial, 640, 480, 15, warmup_frames=15)
    rows: dict[str, list] = {k: [] for k in cards}
    try:
        for _ in range(frames):
            f = cam.grab()
            rgb, depth, K = f["rgb"], f["depth"], f["intrinsics"]
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            got = {int(i): c[0] for i, c in zip([] if ids is None else ids.ravel(), corners)}
            for card, pts in cards.items():
                seen = [i for i in pts if i in got]
                if len(seen) < 2:
                    continue
                obj = np.concatenate([pts[i] for i in seen])
                img = np.concatenate([got[i] for i in seen]).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
                if not ok:
                    continue
                R = cv2.Rodrigues(rvec)[0]
                n, d0 = R[:, 2], float(R[:, 2] @ tvec.ravel())  # the card's plane in the camera: n . X = d0
                mask = np.zeros(depth.shape, np.uint8)
                for i in seen:  # inside each marker (its black and white cells: one plane)
                    cv2.fillConvexPoly(mask, got[i].astype(np.int32), 1)
                v, u = np.nonzero(mask)
                ray = np.linalg.inv(K) @ np.vstack([u, v, np.ones_like(u)])
                z_pnp = d0 / (n @ ray)  # depth along the optical axis where each ray meets the plane
                z = depth[v, u]
                good = (z > 0.05) & (z_pnp > 0.05)
                if good.sum() < 50:
                    continue
                rows[card].append((float(np.median(z_pnp[good])), float(np.median(z[good] / z_pnp[good])),
                                   float(np.median(z[good] - z_pnp[good])), int(good.sum())))  # fmt: skip
    finally:
        cam.close()
    out = []
    for card, r in rows.items():
        if not r:
            continue
        a = np.array(r)
        out.append({"camera": name, "card": card, "frames": len(r), "distance_m": round(float(a[:, 0].mean()), 3),
                    "depth_over_pnp": round(float(np.median(a[:, 1])), 4), "depth_minus_pnp_mm": round(float(np.median(a[:, 2])) * 1000, 1),
                    "spread_mm": round(float(a[:, 2].std()) * 1000, 1), "pixels": int(a[:, 3].mean())})  # fmt: skip
    return out


def main() -> None:
    import yaml
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", nargs="+", default=["wrist", "pole"])
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()
    cfg = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    serials = {c["name"]: str(c["serial"]) for c in cfg["cameras"]}
    for name in a.cameras:
        res = check(name, serials[name], a.frames)
        if not res:
            print(f"{name}: no card in view (base card ids 1, 2 or hand card ids 4-6, two markers or more)")
        for r in res:
            print(f"{r['camera']:5} {r['card']}: {r['distance_m']:.3f} m away; depth reads {r['depth_over_pnp']:.4f} x "
                  f"({r['depth_minus_pnp_mm']:+.1f} mm, spread {r['spread_mm']:.1f}) over {r['frames']} frames, {r['pixels']} px")  # fmt: skip


if __name__ == "__main__":
    main()
