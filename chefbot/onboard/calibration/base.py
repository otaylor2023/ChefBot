"""The base's mocap frame from driving it by hand: Motive's up axis, the marker's heading offset and its place on
the base. Recording only reads Redis; the driving is yours, with the gamepad (bringup/gamepad.py).

    python -m chefbot.onboard.calibration.base record forward.jsonl   # drive ~1 m straight ahead, slowly, then stop
    python -m chefbot.onboard.calibration.base record spin.jsonl      # spin in place, one slow full turn
    python -m chefbot.onboard.calibration.base solve --forward forward.jsonl --spin spin.jsonl [more.jsonl ...]

record logs the raw mocap pose (`chefbot-tidy::pos`, `::ori` as Motive streams it), its tracking flag and freshness
(onboard/mocap.py) at --hz, and the base driver's odometry if it is up, until Ctrl-C or --seconds.

solve, from every live sample:
  - the up axis, two ways that do not depend on each other: the axis the body turns about (spin) and the normal of
    the plane the positions lie in (all files). The floor is the other two axes.
  - marker_yaw_offset_deg: the marker's heading (its body x on the floor) minus the direction the base drove in the
    forward recording. frames.py: base heading = marker heading - offset.
  - marker_xy: the marker's origin in the base frame, from the circle it traces around the base centre in the spin.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from chefbot.onboard import keys as K
from chefbot.onboard.mocap import MocapFreshness

AXES = "xyz"


# -- geometry -------------------------------------------------------------------------


def rotation(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def rotation_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    return (axis / n if n > 1e-9 else np.zeros(3)), angle


def floor_frame(up: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (X, Y, Up) in Motive's frame for a floor frame with z up, right-handed: X is Motive's x unless
    x is up; Y = Up x X. For Motive's Y-up default: X = x, Y = -z."""
    U = np.eye(3)[up]
    X = np.eye(3)[0 if up != 0 else 1]
    return X, np.cross(U, X), U


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# -- recording ----------------------------------------------------------------------


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if "pos" in r]


def live(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["valid"] and r["fresh"] and r["pos"] is not None and r["ori"] is not None]


def record(out: Path, hz: float, seconds: float, redis_client=None) -> None:
    if redis_client is None:
        import redis

        redis_client = redis.Redis(decode_responses=True)
    fresh = MocapFreshness()
    keys = (K.MOCAP_POSITION, K.MOCAP_ORIENTATION, K.MOCAP_VALID, K.BASE_POSE)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    next_print, n, n_live = t0, 0, 0
    print(f"recording {out} (Ctrl-C stops); drive now", flush=True)
    with out.open("w") as f:
        f.write(json.dumps({"header": {"keys": keys, "hz": hz, "started": time.time()}}) + "\n")
        try:
            while not seconds or time.monotonic() - t0 < seconds:
                raw_pos, raw_ori, raw_valid, raw_odom = redis_client.mget(keys)
                age = fresh.update(raw_pos, raw_ori)
                row = {
                    "t": time.monotonic() - t0,
                    "pos": json.loads(raw_pos) if raw_pos else None,
                    "ori": json.loads(raw_ori) if raw_ori else None,  # x, y, z, w as Motive streams it
                    "valid": (raw_valid or "").strip() in ("1", "true"),
                    "fresh": fresh.is_fresh(age),
                    "age_s": age,
                    "odom": json.loads(raw_odom) if raw_odom else None,
                }
                f.write(json.dumps(row) + "\n")
                n += 1
                n_live += bool(row["valid"] and row["fresh"])
                if time.monotonic() >= next_print:
                    p = row["pos"] or [float("nan")] * 3
                    print(f"{row['t']:6.1f}s  {'LIVE ' if row['valid'] and row['fresh'] else 'stale'}  pos {p[0]:7.3f} {p[1]:7.3f} "
                          f"{p[2]:7.3f}  live {n_live}/{n}", flush=True)  # fmt: skip
                    next_print += 0.5
                time.sleep(1.0 / hz)
        except KeyboardInterrupt:
            pass
    print(f"{n} samples, {n_live} live -> {out}")


# -- solving -------------------------------------------------------------------------


def up_from_rotation(rows: list[dict], min_angle_rad: float = math.radians(10)) -> tuple[np.ndarray, int] | None:
    """The world axis the body turned about: mean axis of R_i R_0^T over samples turned more than min_angle."""
    if len(rows) < 2:
        return None
    R0 = rotation(rows[0]["ori"])
    axes = []
    for r in rows[1:]:
        axis, angle = rotation_axis_angle(rotation(r["ori"]) @ R0.T)
        if min_angle_rad < angle < math.pi - min_angle_rad:
            axes.append(axis)
    if not axes:
        return None
    axes = np.array(axes)
    axes *= np.sign(axes @ axes[0])[:, None]  # one sign
    mean = axes.mean(axis=0)
    return mean / np.linalg.norm(mean), len(axes)


def up_from_positions(rows: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """The normal of the plane the positions lie in (least-variance direction), and the three spreads (m)."""
    if len(rows) < 3:
        return None
    P = np.array([r["pos"] for r in rows])
    w, V = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
    return V[:, 0], np.sqrt(np.maximum(w, 0.0))


def nearest_axis(v: np.ndarray) -> tuple[int, float]:
    i = int(np.argmax(np.abs(v)))
    return i, math.degrees(math.acos(min(1.0, abs(float(v[i])))))


def marker_heading(ori, X: np.ndarray, Y: np.ndarray) -> float:
    """The marker body's x axis on the floor (rad, in the floor frame)."""
    bx = rotation(ori)[:, 0]
    return math.atan2(float(bx @ Y), float(bx @ X))


def heading_offset(rows: list[dict], X, Y) -> dict:
    """From a straight forward drive: marker heading minus the direction of travel."""
    P = np.array([[r["pos"] @ X, r["pos"] @ Y] for r in rows])
    d = P[-1] - P[0]
    travel = math.atan2(d[1], d[0])
    headings = np.unwrap([marker_heading(r["ori"], X, Y) for r in rows])
    centered = P - P.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return {
        "distance_m": float(np.linalg.norm(d)),
        "travel_deg": math.degrees(travel),
        "marker_heading_deg": math.degrees(wrap(float(np.mean(headings)))),
        "offset_deg": math.degrees(wrap(float(np.mean(headings)) - travel)),
        "heading_change_deg": math.degrees(float(headings.max() - headings.min())),  # should be ~0 for a straight drive
        "straightness": float(s[1] / max(s[0], 1e-9)),  # sideways scatter / along-track spread; ~0 for a line
    }


def circle_fit(P: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares circle (Kasa): centre, radius, rms residual."""
    A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
    b = (P**2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    residual = float(np.sqrt(np.mean((np.linalg.norm(P - [cx, cy], axis=1) - r) ** 2)))
    return np.array([cx, cy]), r, residual


def marker_xy(rows: list[dict], X, Y, offset_rad: float) -> dict:
    """From a spin in place: the marker origin in the base frame (base x forward), about the circle's centre."""
    P = np.array([[r["pos"] @ X, r["pos"] @ Y] for r in rows])
    centre, radius, residual = circle_fit(P)
    headings = np.array([marker_heading(r["ori"], X, Y) for r in rows]) - offset_rad
    xy = []
    for p, h in zip(P, headings):
        c, s = math.cos(h), math.sin(h)
        v = p - centre
        xy.append([c * v[0] + s * v[1], -s * v[0] + c * v[1]])  # R(-h) v
    xy = np.array(xy)
    turned = float(np.ptp(np.unwrap(headings)))
    return {"marker_xy": xy.mean(axis=0).tolist(), "marker_xy_std_m": xy.std(axis=0).tolist(), "radius_m": radius,
            "circle_rms_m": residual, "turned_deg": math.degrees(turned), "centre": centre.tolist()}  # fmt: skip


def solve(forward: Path | None, spin: Path | None, others: list[Path]) -> dict:
    files = {"forward": forward, "spin": spin} | {f"other{i}": p for i, p in enumerate(others)}
    data = {name: live(load(p)) for name, p in files.items() if p is not None}
    out: dict = {"live_samples": {k: len(v) for k, v in data.items()}}
    everything = [r for rows in data.values() for r in rows]

    rot = up_from_rotation(data.get("spin") or everything)
    pos = up_from_positions(everything)
    if rot is not None:
        i, err = nearest_axis(rot[0])
        out["up_from_rotation"] = {"axis": rot[0].round(4).tolist(), "nearest": AXES[i], "off_deg": err, "samples": rot[1]}
    if pos is not None:
        i, err = nearest_axis(pos[0])
        out["up_from_positions"] = {"normal": pos[0].round(4).tolist(), "nearest": AXES[i], "off_deg": err,
                                    "spread_m": pos[1].round(4).tolist()}  # fmt: skip
    votes = [out[k]["nearest"] for k in ("up_from_rotation", "up_from_positions") if k in out]
    if not votes:
        out["error"] = "not enough live motion to find the up axis"
        return out
    up = AXES.index(votes[0])
    out["up_axis"] = AXES[up]
    out["up_axis_agree"] = len(set(votes)) == 1
    X, Y, _ = floor_frame(up)
    out["floor_frame"] = {"X": X.tolist(), "Y": Y.tolist()}
    offset = 0.0
    if data.get("forward"):
        h = heading_offset(data["forward"], X, Y)
        out["forward"] = h
        offset = math.radians(h["offset_deg"])
    if data.get("spin"):
        out["spin"] = marker_xy(data["spin"], X, Y, offset)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("record")
    a.add_argument("out", type=Path)
    a.add_argument("--hz", type=float, default=100.0)
    a.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: Ctrl-C)")
    b = sub.add_parser("solve")
    b.add_argument("--forward", type=Path)
    b.add_argument("--spin", type=Path)
    b.add_argument("others", type=Path, nargs="*", help="more recordings, for the up axis from positions")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.out, args.hz, args.seconds)
    else:
        print(json.dumps(solve(args.forward, args.spin, args.others), indent=2))


if __name__ == "__main__":
    main()
