"""Renders of a checked asset, for the judge and for people (MuJoCo's renderer: what a robot's cameras would see).

    sheet.jpg        six views: on the counter (three-quarter, top over a 5 cm grid, side), the tips, the hinge end,
                     and squeezed shut under force
    actuation.jpg    the squeeze as a strip of frames, with the force arrows at grip_a/grip_b and the force in N
    actuation.mp4    the same, the whole squeeze and release
    top_mask.npy     the top view's silhouette (for diversity)

Uses the traces checks.py left in the asset's lab/ folder (settled.npz, squeeze.npz).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chefbot.autoresearch.generator_lab import scene as S

W, H = 640, 480
ARROW_RGBA = np.array([0.9, 0.1, 0.1, 1.0], np.float32)


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _caption(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = _font(20)
    dr.rectangle([0, 0, im.width, 30], fill=(255, 255, 255))
    dr.text((8, 4), text, fill=(20, 20, 20), font=f)
    return im


def _object_box(m, d) -> tuple[np.ndarray, np.ndarray]:
    obj = S.subtree(m, m.body("object").id)
    return S.world_box(m, d, [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in obj])


def _camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def _opt() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [0, 1, 1, 0, 0, 0]  # visual geoms (groups 1, 2): the object's, and the lab's counter (_models)
    opt.sitegroup[:] = 0
    return opt


def _render(r: mujoco.Renderer, m, d, cam, arrows=()) -> np.ndarray:
    opt = _opt()
    r.update_scene(d, cam, opt)
    for p, direction in arrows:  # (point, unit direction * length)
        if r.scene.ngeom >= r.scene.maxgeom:
            break
        g = r.scene.geoms[r.scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), ARROW_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, p - direction, p)
        r.scene.ngeom += 1
    return r.render()


def _models(asset_dir: Path, R: np.ndarray, pos, fixed: bool, grid: bool = False):
    xml = S.compose(asset_dir, fixed=fixed, pos=pos, R=R, gravity=False, grid=grid)
    # Put the lab's geoms in group 2 so the visual filter (groups 1, 2) shows them while hiding the object's
    # collision geoms (groups 0 and 3).
    xml = xml.replace('<geom name="lab_floor"', '<geom group="2" name="lab_floor"').replace('<geom name="lab_counter"', '<geom group="2" name="lab_counter"').replace(
        '<geom name="lab_grid"', '<geom group="2" name="lab_grid"')
    return S.load(xml)


def render_all(asset_dir: Path, lab_dir: Path | None = None, title: str = "", video: bool = True) -> dict:
    asset_dir = Path(asset_dir)
    lab_dir = Path(lab_dir or asset_dir / "lab")
    out: dict = {}
    settled = np.load(lab_dir / "settled.npz")
    m, d = _models(asset_dir, settled["R"], settled["pos"], fixed=False)
    mg, dg = _models(asset_dir, settled["R"], settled["pos"], fixed=False, grid=True)
    for mm, dd in ((m, d), (mg, dg)):
        dd.qpos[:] = settled["qpos"]
        mujoco.mj_forward(mm, dd)
    lo, hi = _object_box(m, d)
    c, L = (lo + hi) / 2, float(np.max(hi - lo))
    tips = (d.site_xpos[m.site("tip_a").id] + d.site_xpos[m.site("tip_b").id]) / 2
    grips = (d.site_xpos[m.site("grip_a").id] + d.site_xpos[m.site("grip_b").id]) / 2
    u = (tips - grips) / max(np.linalg.norm(tips - grips), 1e-6)  # along the arms, toward the tips
    span = float(np.abs((hi - lo) @ u))  # the object's extent along the arms (near enough: it lies along them)
    back = c - u * (span / 2 - 0.04)  # 4 cm in from the hinge end
    views = []
    with mujoco.Renderer(m, H, W) as r:
        views.append(_caption(_render(r, m, d, _camera(c, 0.55 * L + 0.25, 135, -32)), "on the counter"))
        views.append(_caption(_render(r, m, d, _camera(c, 0.55 * L + 0.2, 90, -4)), "side"))
        views.append(_caption(_render(r, m, d, _camera(tips, 0.16, 150, -38)), "tips"))
        views.append(_caption(_render(r, m, d, _camera(back, 0.16, 20, -38)), "hinge end"))
    with mujoco.Renderer(mg, H, W) as r:
        top = _render(r, mg, dg, _camera(c, 0.75 * L + 0.12, 90, -89.9))
        views.insert(1, _caption(top, "top, 5 cm grid"))
        # A silhouette from the same camera at a fixed scale (for diversity): the object's geoms only.
        seg = mujoco.Renderer(mg, 160, 160)
        seg.enable_segmentation_rendering()
        seg.update_scene(dg, _camera(c, 0.75, 90, -89.9), _opt())
        ids = seg.render()[..., 0]
        objs = S.subtree(mg, mg.body("object").id)
        mask = np.isin(ids, [g for g in range(mg.ngeom) if int(mg.geom_bodyid[g]) in objs])
        np.save(lab_dir / "top_mask.npy", mask)
        seg.close()

    # The squeeze, from the stand.
    sq = np.load(lab_dir / "squeeze.npz")
    ms, ds = _models(asset_dir, sq["R"], sq["pos"], fixed=True)
    forces = sq["force"]
    fmax = float(forces.max())
    # Strip frames by how far the tips have closed: rest, 2/3 and 1/3 open, first closed, hardest squeeze, released.
    gaps = []
    for q in sq["qpos"]:
        ds.qpos[:] = q
        mujoco.mj_kinematics(ms, ds)
        gaps.append(float(np.linalg.norm(ds.site_xpos[ms.site("tip_b").id] - ds.site_xpos[ms.site("tip_a").id])))
    gaps = np.array(gaps)
    peak = int(np.argmax(forces))
    rest, closed = gaps[0], float(gaps[: peak + 1].min())
    frames_idx = {0, peak, len(forces) - 1}
    for frac in (2 / 3, 1 / 3, 0.0):
        target = closed + frac * (rest - closed)
        frames_idx.add(int(np.argmax(gaps[: peak + 1] <= target + 1e-4)))
    frames_idx = sorted(frames_idx)
    cam_s = None
    strip, movie = [], []
    with mujoco.Renderer(ms, H, W) as r:
        idx_all = list(range(0, len(forces), 3)) if video else []
        for i in sorted(set(idx_all) | set(frames_idx)):
            ds.qpos[:] = sq["qpos"][i]
            mujoco.mj_forward(ms, ds)
            if cam_s is None:
                lo2, hi2 = _object_box(ms, ds)
                cam_s = _camera((lo2 + hi2) / 2, 0.6 * float(np.max(hi2 - lo2)) + 0.18, 115, -55)
            f = float(forces[i])
            arrows = []
            if f > 0:
                ga, gb = sq["grip_a"][i], sq["grip_b"][i]
                n = (gb - ga) / max(np.linalg.norm(gb - ga), 1e-9)
                length = 0.02 + 0.06 * f / fmax
                arrows = [(ga, n * length), (gb, -n * length)]
            img = _render(r, ms, ds, cam_s, arrows)
            label = f"squeeze {f:4.1f} N at each grip" if f > 0 else ("at rest" if i == 0 else "released")
            if i in frames_idx:
                strip.append(_caption(img, label))
            if i in idx_all:
                movie.append(np.asarray(_caption(img, f"t {sq['t'][i]:.2f} s   {label}")))
    views.append(strip[-3] if len(strip) >= 3 else strip[-1])  # just closed

    sheet = Image.new("RGB", (3 * W, 2 * H + 40), "white")
    for k, v in enumerate(views[:6]):
        sheet.paste(v, ((k % 3) * W, 40 + (k // 3) * H))
    ImageDraw.Draw(sheet).text((10, 8), title, fill=(0, 0, 0), font=_font(24))
    sheet.save(lab_dir / "sheet.jpg", quality=88)
    n = len(strip)
    act = Image.new("RGB", (n * W // 2, H // 2), "white")
    for k, v in enumerate(strip):
        act.paste(v.resize((W // 2, H // 2)), (k * W // 2, 0))
    act.save(lab_dir / "actuation.jpg", quality=88)
    out.update(sheet=str(lab_dir / "sheet.jpg"), actuation=str(lab_dir / "actuation.jpg"))
    if video and movie:
        import imageio.v3 as iio

        iio.imwrite(lab_dir / "actuation.mp4", np.stack(movie), fps=33, codec="libx264", quality=7)
        out["video"] = str(lab_dir / "actuation.mp4")
    return out
