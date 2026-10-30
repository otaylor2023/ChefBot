"""Renders of a generated building, filmed during a scripted trip (faults off): the cart calls a car at the lowest
floor, rides to the highest and drives out. MuJoCo's renderer: what a robot's cameras would see.

    sheet.jpg        six views: the origin hall as the car opens, the hall call panel, the car interior with the cart,
                     the car's panel, the car interior mid-ride, the destination hall on arrival
    actuation.jpg    the trip in six frames (call, doors open, cart in, riding, arrived, cart out)
    actuation.mp4    a time-lapse: origin hall | car interior | destination hall, with the clock and the last event
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chefbot.environments.mujoco import scene as SC
from chefbot.generators.elevator.checks import Ctx, _approach, _press_until_registered

W, H = 640, 480


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(img: np.ndarray, text: str, size: int = 20) -> Image.Image:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, size + 10], fill=(255, 255, 255))
    dr.text((6, 3), text, fill=(20, 20, 20), font=_font(size))
    return im


def _cam(eye, look) -> mujoco.MjvCamera:
    eye, look = np.asarray(eye, float), np.asarray(look, float)
    f = look - eye
    dist = float(np.linalg.norm(f))
    f /= max(dist, 1e-9)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
    cam.elevation = math.degrees(math.asin(np.clip(f[2], -1, 1)))
    return cam


class Film:
    def __init__(self, sc, ctx: Ctx, car: str, a: str, b: str):
        self.sc, self.ctx, self.car, self.a, self.b = sc, ctx, car, a, b
        m = sc.m
        m.vis.map.znear = 0.02 / max(m.stat.extent, 1e-6)  # clip 2 cm from the camera, whatever the building's size
        m.vis.map.zfar = 200.0 / max(m.stat.extent, 1e-6)
        m.vis.global_.fovy = 70.0  # wide, like a robot's camera in a small car
        self.r = mujoco.Renderer(sc.m, H, W)
        self.small = mujoco.Renderer(sc.m, 240, 320)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0
        self.frames: list[np.ndarray] = []
        self.next_t = 0.0
        self.event = ""

    def hall(self, floor: str):
        p, R = self.sc.site(self.ctx.obj[self.car]["landings"][floor]["sill_site"])
        return _cam(p - R[:, 0] * 3.4 + R[:, 1] * 0.9 + [0, 0, 1.7], p + [0, 0, 1.1])

    def inside(self):
        pf, _ = self.sc.site(self.ctx.obj[self.car]["floor_site"])
        ps, Rs = self.sc.site(self.ctx.obj[self.car]["sill_site"])
        depth = float(np.linalg.norm((pf - ps)[:2]))
        back = pf - Rs[:, 0] * depth * 0.8 + Rs[:, 1] * depth * 0.45  # a back corner
        return _cam(back + [0, 0, 1.8], (pf + ps) / 2 + [0, 0, 0.5])

    def closeup(self, button: str):
        p, R = self.sc.site(self.ctx.obj[button]["site"])
        return _cam(p + R[:, 2] * 0.5 + [0, 0, 0.08], p)

    def shot(self, cam, small: bool = False) -> np.ndarray:
        r = self.small if small else self.r
        r.update_scene(self.sc.d, cam, self.opt)
        return r.render()

    def tick(self, sc) -> None:
        if sc.sim.t + 1e-9 < self.next_t:
            return
        self.next_t = sc.sim.t + 0.25
        ev = [e for e in sc.sim.timeline if e["t"] > sc.sim.t - 0.3]
        if ev:
            self.event = ev[-1]["type"] + (" " + str(ev[-1].get("button") or ev[-1].get("floor") or "")).rstrip()
        panels = [self.shot(self.hall(self.a), True), self.shot(self.inside(), True), self.shot(self.hall(self.b), True)]
        frame = np.concatenate(panels, axis=1)
        self.frames.append(np.asarray(_caption(frame, f"t {sc.sim.t:6.1f} s   {self.event}", 16)))

    def close(self) -> None:
        self.r.close()
        self.small.close()


def render_all(asset_dir: Path, lab_dir: Path | None = None, title: str = "", video: bool = True, sandbox=(), python=None) -> dict:
    asset_dir = Path(asset_dir)
    lab_dir = Path(lab_dir or asset_dir / "lab")
    lab_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads((asset_dir / "params.json").read_text())
    from chefbot.autoresearch.generator_lab import family as F

    ctx = Ctx(asset_dir, F.load("elevator"), params, sandbox, python)
    a, b = ctx.floors[0], ctx.floors[-1]
    car = next(c for c in ctx.cars if a in ctx.obj[c]["landings"] and b in ctx.obj[c]["landings"])
    sc = ctx.scenario()
    film = Film(sc, ctx, car, a, b)
    if video:
        sc.on_tick = film.tick
    views, strip = {}, []
    call = ctx.call_button(a, b)
    car_btn = (ctx.buttons(type="car_floor", elevator=car, floor=b) or [None])[0]
    start, za = _approach(sc, ctx.obj[car]["landings"][a]["sill_site"])
    sc.place_cart(start[:2], 0.0, za)
    sc.run(0.5)
    views["call panel"] = film.shot(film.closeup(call)) if call else None
    if car_btn:
        views["car panel"] = film.shot(film.closeup(car_btn))
    strip.append(("calling", film.shot(film.hall(a))))
    _press_until_registered(sc, call)
    tol = float(ctx.run_params["leveling_tol_m"]) + 0.001
    sc.run(float(ctx.run_params["max_wait_s"]) + 10, until=lambda: sc.at_floor(a, tol) is not None)
    views["the car opens"] = film.shot(film.hall(a))
    strip.append(("doors open", views["the car opens"]))
    sc.drive_cart(sc.site(ctx.obj[car]["floor_site"])[0][:2], timeout_s=25)
    views["in the car"] = film.shot(film.inside())
    strip.append(("cart in", views["in the car"]))
    if car_btn:
        _press_until_registered(sc, car_btn)
    moving = sc.run(30, until=lambda: abs(sc.joint_v(ctx.obj[car]["joint"])) > 0.5)
    if moving is not None:
        sc.run(1.0)
        views["riding"] = film.shot(film.inside())
        strip.append(("riding", views["riding"]))
    sc.run(float(ctx.run_params["max_wait_s"]) + 60, until=lambda: sc.at_floor(b, tol) == car)
    views["arrival"] = film.shot(film.hall(b))
    strip.append(("arrived", views["arrival"]))
    out, zb = _approach(sc, ctx.obj[car]["landings"][b]["sill_site"])
    sc.drive_cart(out[:2], timeout_s=25, floor_z=zb)
    strip.append(("cart out", film.shot(film.hall(b))))
    sc.run(1.0)
    film.close()
    sc.close()

    order = ["the car opens", "call panel", "in the car", "car panel", "riding", "arrival"]
    tiles = [_caption(views[k], k) if views.get(k) is not None else Image.new("RGB", (W, H), "white") for k in order]
    sheet = Image.new("RGB", (3 * W, 2 * H + 40), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 3) * W, 40 + (i // 3) * H))
    ImageDraw.Draw(sheet).text((10, 8), title, fill=(0, 0, 0), font=_font(24))
    sheet.save(lab_dir / "sheet.jpg", quality=88)
    act = Image.new("RGB", (len(strip) * W // 2, H // 2), "white")
    for i, (label, img) in enumerate(strip):
        act.paste(_caption(img, label).resize((W // 2, H // 2)), (i * W // 2, 0))
    act.save(lab_dir / "actuation.jpg", quality=88)
    res = {"sheet": str(lab_dir / "sheet.jpg"), "actuation": str(lab_dir / "actuation.jpg")}
    if video and film.frames:
        import imageio.v3 as iio

        iio.imwrite(lab_dir / "actuation.mp4", np.stack(film.frames), fps=12, codec="libx264", quality=7)
        res["video"] = str(lab_dir / "actuation.mp4")
    return res
