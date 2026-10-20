"""Animate a planned chunk on its plots: the gripper (A, B, F and its outline)
moving along the waypoints as the simulator follows them (straight lines,
orientation interpolated), with the path drawn behind it.

    python -m experiments.ortho_grasp.animate experiments/ortho_grasp/results/chunk_claude/s05_0 [...]

Writes plan.mp4 next to the run's waypoints.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.ortho_grasp.chunk import _slerp_poses
from experiments.ortho_grasp.frames import FLANGE
from experiments.ortho_grasp.render.plots import PlotSpec

FPS = 20
SPEED = 0.10  # m/s of the grip point, for pacing
HOLD = 0.6  # s held at each waypoint
COLORS = {"A": (224, 32, 26), "B": (26, 95, 224), "F": (240, 140, 0), "body": (60, 60, 60), "trail": (24, 163, 58)}


def _font(size: int):
    for name in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def timeline(start: dict, waypoints: list[dict]) -> list[tuple[np.ndarray, float, str, int]]:
    """(pose, width, gripper state, waypoint being approached) per frame."""
    from experiments.ortho_grasp.resolve import resolve_p1

    s = {k: np.asarray(v) for k, v in start.items()}
    p0 = resolve_p1({"A": s["A0"], "B": s["B0"], "F": s["F0"]}, {}).pose
    poses = [p0] + [np.asarray(w["pose"]) for w in waypoints]
    widths = [0.085] + [float(w["width"]) for w in waypoints]
    states = ["open"] + [w["gripper"] for w in waypoints]
    frames = [(p0, widths[0], "open", 0)] * int(HOLD * FPS)
    closed = False
    for i in range(len(poses) - 1):
        dist = float(np.linalg.norm(poses[i + 1][:3, 3] - poses[i][:3, 3]))
        n = max(int(dist / SPEED * FPS), 6)
        for p in _slerp_poses(poses[i], poses[i + 1], n)[1:]:
            frames.append((p, widths[i + 1] if not closed else widths[i + 1], "closed" if closed else "open", i + 1))
        if states[i + 1] == "closed" and not closed:
            closed = True
        frames += [(poses[i + 1], widths[i + 1], "closed" if closed else "open", i + 1)] * int(HOLD * FPS)
    return frames


def gripper_points(pose: np.ndarray, width: float, closed: bool) -> dict[str, np.ndarray]:
    m, x, z = pose[:3, 3], pose[:3, 0], pose[:3, 2]
    half = 0.012 if closed else width / 2
    return {"A": m - x * half, "B": m + x * half, "F": m - z * FLANGE, "M": m}


def animate(run: Path, inputs: Path, out: Path | None = None) -> Path:
    waypoints = json.loads((run / "waypoints.json").read_text())
    start = json.loads((inputs / "start.json").read_text())
    views = {v: (PlotSpec.load(inputs / f"{v}.json"), Image.open(inputs / f"{v}.png").convert("RGB")) for v in ("top", "side")}
    font, small = _font(26), _font(20)
    trail: list[np.ndarray] = []
    frames = []
    for pose, width, state, target in timeline(start, waypoints):
        pts = gripper_points(pose, width, state == "closed")
        trail.append(pts["M"])
        panels = []
        for v, (spec, bg) in views.items():
            img = bg.copy()
            d = ImageDraw.Draw(img)
            px = lambda p: tuple(spec.to_pixel(spec.project(np.asarray(p)[None]))[0])  # noqa: E731
            if len(trail) > 1:
                d.line([px(p) for p in trail], fill=COLORS["trail"], width=3)
            a, b, f, m = px(pts["A"]), px(pts["B"]), px(pts["F"]), px(pts["M"])
            d.line([a, b], fill=COLORS["body"], width=5)
            d.line([m, f], fill=COLORS["body"], width=5)
            for name in ("A", "B", "F"):
                x, y = px(pts[name])
                d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=COLORS[name], outline=(255, 255, 255), width=2)
                d.text((x + 10, y - 26), name, fill=COLORS[name], font=font)
            panels.append(np.asarray(img)[: spec.top + spec.area + 70, :])
        frame = np.concatenate(panels, axis=1)
        label = f"to waypoint {target}  ·  gripper {state}" if target else "waypoint 0 (start)  ·  gripper open"
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text((20, im.height - 46), label, fill=(20, 20, 20), font=small)
        frames.append(np.asarray(im)[::2, ::2])
    h, w = frames[0].shape[:2]
    frames = [f[: h - h % 2, : w - w % 2] for f in frames]
    out = out or run / "plan.mp4"
    iio.imwrite(out, np.stack(frames), fps=FPS, codec="libx264")
    return out


def main() -> int:
    """Paths of chunk run folders; --synced: the plots and the simulator side by side (needs exec_track.json)."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for arg in args:
        run = Path(arg)
        scene = run.name.split("_")[0]
        inputs = run.parents[1] / "inputs" / "chunk" / scene
        print(synced(run, inputs) if "--synced" in sys.argv else animate(run, inputs))
    return 0



def synced(run: Path, inputs: Path, out: Path | None = None) -> Path:
    """The plots and the simulator side by side, frame for frame: each plot frame
    draws the gripper where the robot really is at that sim frame (from
    exec_track.json), over the model's planned path (gray, dashed)."""
    waypoints = json.loads((run / "waypoints.json").read_text())
    start = json.loads((inputs / "start.json").read_text())
    track = json.loads((run / "exec_track.json").read_text())
    sim = iio.imread(run / "exec.mp4")
    n = min(len(track), len(sim))
    views = {v: (PlotSpec.load(inputs / f"{v}.json"), Image.open(inputs / f"{v}.png").convert("RGB")) for v in ("top", "side")}
    font, small = _font(26), _font(22)
    s = {k: np.asarray(v) for k, v in start.items()}
    plan = [(s["A0"] + s["B0"]) / 2] + [np.asarray(w["pose"])[:3, 3] for w in waypoints]
    trail: list[np.ndarray] = []
    frames = []
    for i in range(n):
        pose = np.asarray(track[i]["pose"])
        width = track[i]["width"]
        m, x, z = pose[:3, 3], pose[:3, 0], pose[:3, 2]
        pts = {"A": m - x * width / 2, "B": m + x * width / 2, "F": m - z * FLANGE, "M": m}
        trail.append(m)
        panels = []
        for v, (spec, bg) in views.items():
            img = bg.copy()
            d = ImageDraw.Draw(img)
            px = lambda p: tuple(spec.to_pixel(spec.project(np.asarray(p)[None]))[0])  # noqa: E731
            plan_px = [px(p) for p in plan]
            for a, b in zip(plan_px, plan_px[1:]):  # dashed planned path
                seg = np.linspace(a, b, 24)
                for k in range(0, len(seg) - 1, 2):
                    d.line([tuple(seg[k]), tuple(seg[k + 1])], fill=(140, 140, 140), width=3)
            for k, p in enumerate(plan_px[1:], start=1):
                d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], outline=(110, 110, 110), width=3)
                d.text((p[0] - 26, p[1] + 6), str(k), fill=(90, 90, 90), font=small)
            if len(trail) > 1:
                d.line([px(p) for p in trail], fill=COLORS["trail"], width=4)
            a, b, f, mm = px(pts["A"]), px(pts["B"]), px(pts["F"]), px(pts["M"])
            d.line([a, b], fill=COLORS["body"], width=5)
            d.line([mm, f], fill=COLORS["body"], width=5)
            for name in ("A", "B", "F"):
                xx, yy = px(pts[name])
                d.ellipse([xx - 8, yy - 8, xx + 8, yy + 8], fill=COLORS[name], outline=(255, 255, 255), width=2)
                d.text((xx + 10, yy - 26), name, fill=COLORS[name], font=font)
            panels.append(np.asarray(img)[spec.top - 40 : spec.top + spec.area + 70, : spec.left + spec.area + 10])
        plots_row = np.concatenate(panels, axis=1)
        h = plots_row.shape[0]
        frame_sim = np.asarray(Image.fromarray(sim[i]).resize((round(sim[i].shape[1] * h / sim[i].shape[0]), h)))
        row = np.concatenate([plots_row, frame_sim], axis=1)
        im = Image.fromarray(row)
        ImageDraw.Draw(im).text((20, im.height - 34), f"t = {track[i]['t'] - track[0]['t']:4.1f} s   gripper {1000 * width:.0f} mm open",
                                fill=(20, 20, 20), font=small)  # fmt: skip
        frames.append(np.asarray(im)[::2, ::2])
    hh, ww = frames[0].shape[:2]
    frames = [f[: hh - hh % 2, : ww - ww % 2] for f in frames]
    out = out or run / "synced.mp4"
    iio.imwrite(out, np.stack(frames), fps=10, codec="libx264")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
