"""Pictures of the explored kitchens, for the exploration report.

For each layout: a top-down render with fixture names, the loose objects and the robot's start
pose; two angled renders as a person would see the kitchen; and a 2D plan from the start map
(fixtures, cabinets and drawers as footprints, objects as dots). Objects come from the same task
and seed as the runs.

    python -m chefbot.explore.layout_views --out docs/exploration_report/media/layouts \
        1:chefbot/tasks/explore_kitchen.yaml 7:chefbot/tasks/explore_kitchen_hard.yaml ...
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

UPPER_Z = 1.4  # m: kitchen geoms centered above this (wall cabinets, hoods, walls, lights) are left out from above
PX_PER_M = 120  # top-down and plan scale
MARGIN = 0.6  # m around the fixtures
WHITE, BLACK = (255, 255, 255), (20, 20, 20)
OBJECT, ROBOT = (230, 60, 60), (40, 110, 230)
FIXTURE_FILL = (205, 205, 205)
STORAGE_COLORS = {"low": (200, 160, 90), "drawer": (70, 130, 200), "high": (130, 80, 30)}


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=WHITE, outline=BLACK) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=3, stroke_fill=outline, anchor="mm")


class Frame:
    """World xy to pixels: x to the right, y up."""

    def __init__(self, lo, hi, px_per_m: float = PX_PER_M):
        self.lo, self.hi, self.s = np.asarray(lo, float), np.asarray(hi, float), px_per_m
        self.size = tuple(int(round(v)) for v in (self.hi - self.lo) * self.s)

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.lo[0]) * self.s, (self.hi[1] - y) * self.s


def _bounds(start: dict) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for f in start["fixtures"].values():
        points += _corners(f) if f.get("size") else [tuple(f["location"])]
    points = np.array(points)
    return points.min(axis=0) - MARGIN, points.max(axis=0) + MARGIN


def _corners(f: dict) -> list[tuple[float, float]]:
    c = np.asarray(f["location"], float)
    w, d = f.get("size") or [0.3, 0.3]
    fx, fy = f.get("facing") or [1.0, 0.0]
    along, out = np.array([-fy, fx]), np.array([fx, fy])
    return [tuple(c + a * w / 2 * along + b * d / 2 * out) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _is_storage(name: str) -> bool:
    return name.startswith(("cabinet", "drawer", "shelves"))


def _fixture_labels(draw, frame: Frame, fixtures: dict, font) -> None:
    """Counters and appliances; the stove's burners are left to the stove."""
    for name, f in fixtures.items():
        if not _is_storage(name) and not name.startswith("burner"):
            _label(draw, frame.px(*f["location"]), name, font)


def _robot_arrow(draw, frame: Frame, pose, font) -> None:
    x, y, yaw = pose
    p, tip = frame.px(x, y), frame.px(x + 0.45 * math.cos(yaw), y + 0.45 * math.sin(yaw))
    r = 0.3 * frame.s
    draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=ROBOT, width=4)
    draw.line([p, tip], fill=ROBOT, width=6)
    _label(draw, (p[0], p[1] + r + 14), "robot start", font, fill=ROBOT, outline=WHITE)


def _objects(draw, frame: Frame, objects: dict, font) -> list[str]:
    """Numbered dots; returns the legend lines, in number order."""
    lines = []
    for i, (name, pos) in enumerate(sorted(objects.items()), 1):
        p = frame.px(pos[0], pos[1])
        draw.ellipse([p[0] - 11, p[1] - 11, p[0] + 11, p[1] + 11], fill=OBJECT, outline=WHITE, width=2)
        draw.text(p, str(i), font=font, fill=WHITE, anchor="mm")
        lines.append(f"{i}  {name}  (z {pos[2]:.2f} m)")
    return lines


def _compose(img: Image.Image, caption: str, legend: list[str], font) -> Image.Image:
    """The picture with the object legend on its right and a caption below."""
    w, h = img.size
    pw = 290
    lines = caption.split("; ")
    canvas = Image.new("RGB", (w + pw, max(h, 60 + 24 * len(legend)) + 14 + 24 * len(lines)), WHITE)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((w + 16, 16), "objects", font=font, fill=BLACK)
    for i, line in enumerate(legend):
        draw.text((w + 16, 46 + 24 * i), line, font=font, fill=OBJECT)
    for i, line in enumerate(lines):
        draw.text((10, canvas.size[1] - 24 * (len(lines) - i) - 6), line, font=font, fill=BLACK)
    return canvas


def plan(start: dict, objects: dict, pose, title: str) -> Image.Image:
    lo, hi = _bounds(start)
    frame = Frame(lo, hi)
    img = Image.new("RGB", frame.size, (250, 250, 248))
    draw = ImageDraw.Draw(img)
    mid = _font(17)
    fixtures = start["fixtures"]
    for name, f in fixtures.items():
        if not _is_storage(name) and f.get("size"):
            draw.polygon([frame.px(*c) for c in _corners(f)], fill=FIXTURE_FILL, outline=(120, 120, 120))
    counts = {"low": 0, "drawer": 0, "high": 0}
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for name, f in fixtures.items():
        if not _is_storage(name) or not f.get("size"):
            continue
        z0 = (f.get("z") or [0.0])[0]
        kind = "drawer" if name.startswith("drawer") else "high" if z0 > 1.2 else "low"
        counts[kind] += 1
        pts = [frame.px(*c) for c in _corners(f)]
        if kind == "high":
            od.polygon(pts, outline=STORAGE_COLORS[kind] + (255,), width=2)
        else:
            od.polygon(pts, fill=STORAGE_COLORS[kind] + (90,), outline=STORAGE_COLORS[kind] + (255,), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    _fixture_labels(draw, frame, fixtures, mid)
    legend = _objects(draw, frame, objects, mid)
    _robot_arrow(draw, frame, pose, mid)
    storage = (f"{counts['low']} low cabinets (beige), {counts['drawer']} drawers (blue), "
               f"{counts['high']} wall cabinets (brown outline)") if sum(counts.values()) else "no cabinets or drawers on the map"  # fmt: skip
    return _compose(img, f"{title}, plan from the start map: grey counters and appliances, {storage}; "
                    "red numbered objects, blue robot start", legend, mid)  # fmt: skip


class Renderer:
    def __init__(self, env, width: int, height: int):
        from chefbot.envs.robocasa.detector import scene_objects

        self.env, self.model, self.data = env, env.model, env.data
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.r = mujoco.Renderer(self.model, height, width)
        objects = {b for _, _, b in scene_objects(env)}
        self.keep = {g for g in range(self.model.ngeom)
                     if env.is_robot_body(int(self.model.geom_bodyid[g]))
                     or _root(self.model, int(self.model.geom_bodyid[g])) in objects}  # fmt: skip
        self.walls = {g for g in range(self.model.ngeom)
                      if env.root_body_name(int(self.model.geom_bodyid[g])).startswith("wall")}  # fmt: skip

    def shot(self, lookat, distance, azimuth, elevation, ortho_extent: float | None = None, hide_upper=False,
             cut_walls: bool = False):  # fmt: skip
        """`cut_walls`: leave out the walls on the camera's side of `lookat`, which would hide the room."""
        vis = self.model.vis.global_
        vis.orthographic = 1 if ortho_extent else 0
        vis.fovy = ortho_extent or 45.0
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        self.r.update_scene(self.data, cam, opt)
        toward = -np.array([math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))])
        scn = self.r.scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM or g.objid in self.keep:
                continue
            if (hide_upper and g.pos[2] > UPPER_Z) or (
                    cut_walls and g.objid in self.walls and (np.asarray(g.pos[:2]) - lookat[:2]) @ toward > 0.3):
                g.pos[2] = -50.0
        return Image.fromarray(self.r.render())


def _root(model, body: int) -> int:
    while model.body_parentid[body] != 0:
        body = int(model.body_parentid[body])
    return body


def render_layout(layout: int, task: Path, out: Path, seed: int = 0) -> list[Path]:
    from chefbot.envs.robocasa.build_map import STORAGE_START_MAP_FILE, START_MAP_FILE, object_positions
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv
    from chefbot.envs.symbolic.scenario import load_yaml

    spec = load_yaml(task).get("robocasa") or {}
    style = int(spec.get("style", 1))
    config = RoboCasaConfig.load(layout=layout, style=style, seed=seed)
    if spec.get("objects"):
        config = dataclasses.replace(config, objects=tuple(dict(o) for o in spec["objects"]))
    start_file = STORAGE_START_MAP_FILE if spec.get("storage") else START_MAP_FILE
    start = yaml.safe_load((config.map_dir / start_file).read_text())
    if not spec.get("storage") and (config.map_dir / STORAGE_START_MAP_FILE).exists():
        start = yaml.safe_load((config.map_dir / STORAGE_START_MAP_FILE).read_text())  # the plan shows them anyway
    env = RoboCasaEnv(config, render=False)
    try:
        objects = {k: v["center"] for k, v in object_positions(env).items()}
        state = env.get_state()
        pose = (state.base_pose.x, state.base_pose.y, state.base_pose.yaw)
        tag = f"layout{layout:03d}"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        title = f"Layout {layout}"

        lo, hi = _bounds(start)
        frame = Frame(lo, hi)
        w, h = frame.size
        r = Renderer(env, w, h)
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.9])
        top = r.shot(center, 12.0, 90.0, -90.0, ortho_extent=hi[1] - lo[1], hide_upper=True)
        draw = ImageDraw.Draw(top)
        font = _font(17)
        _fixture_labels(draw, frame, start["fixtures"], font)
        legend = _objects(draw, frame, objects, font)
        _robot_arrow(draw, frame, pose, font)
        canvas = _compose(top, f"{title}, seen from above (wall cabinets and walls left out): "
                          "white fixture names, red numbered objects, blue robot start", legend, font)  # fmt: skip
        paths.append(out / f"{tag}_top.png")
        canvas.save(paths[-1])

        r2 = Renderer(env, 1280, 800)
        span = float(np.max(hi - lo))
        for i, azimuth in enumerate((60.0, 240.0)):
            img = r2.shot(center, span * 1.05, azimuth, -38.0, cut_walls=True)
            paths.append(out / f"{tag}_view{i + 1}.png")
            img.save(paths[-1])

        paths.append(out / f"{tag}_plan.png")
        plan(start, objects, pose, title).save(paths[-1])
        return paths
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("layouts", nargs="+", help="layout:task.yaml")
    args = ap.parse_args()
    for item in args.layouts:
        layout, task = item.split(":", 1)
        for p in render_layout(int(layout), Path(task), args.out):
            print(p)


if __name__ == "__main__":
    main()
