"""The lab's MuJoCo scenes around a generated object (contract.md): the object's model.xml composed into a scene
with a counter, lights and test objects, either fixed in space (a stand: for squeezing and pinching) or free (dropped
on the counter). Physics options match ChefBot's RoboCasa env (robosuite's base.xml, timestep 0.002, noslip 4), so
what passes here behaves the same in a kitchen.

Scene names all start with `lab_`; the object's own names are kept as they are (its sites and bodies are found by
their plain names).
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

COUNTER_TOP = 0.92  # m, a kitchen counter
ALLOWED = {"asset", "default", "worldbody", "equality", "contact", "tendon", "compiler"}
OPTIONS = '<option timestep="0.002" impratio="20" cone="elliptic" density="1.2" viscosity="0.00002" noslip_iterations="4"/>'
VISUAL = """<visual>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <map znear="0.01" zfar="20" force="0.02"/>
    <scale forcewidth="0.02" contactwidth="0.01" contactheight="0.004"/>
  </visual>"""
ASSETS = """
    <texture name="lab_sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.62 0.65 0.70" width="256" height="256"/>
    <texture name="lab_grid" type="2d" builtin="checker" rgb1="0.62 0.62 0.60" rgb2="0.42 0.42 0.40" width="512" height="512"/>
    <material name="lab_grid" texture="lab_grid" texrepeat="20 20" texuniform="true"/>
    <material name="lab_counter" rgba="0.86 0.85 0.82 1" reflectance="0.08" specular="0.3"/>
    <material name="lab_floor" rgba="0.45 0.43 0.40 1"/>
    <material name="lab_test" rgba="0.85 0.55 0.25 1"/>"""


class AssetError(ValueError):
    """The generated model.xml breaks the contract (or does not compile)."""


@dataclass
class Frame:
    """The object's natural axes, in its own frame: u along the arms (toward the tips), n the opening direction
    (tip_a toward tip_b at rest), h = u x n (the hinge axis). R maps the object frame to the lab's: u->x, n->y, h->z."""

    u: np.ndarray
    n: np.ndarray
    h: np.ndarray

    @property
    def R(self) -> np.ndarray:
        return np.stack([self.u, self.n, self.h])  # rows: world x, y, z in object coordinates


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def _resolve_files(root: ET.Element, asset_dir: Path) -> None:
    comp = root.find("compiler")
    meshdir = asset_dir / (comp.get("meshdir", "") if comp is not None else "")
    texdir = asset_dir / (comp.get("texturedir", "") if comp is not None else "")
    asset = root.find("asset")
    for el in asset.iter() if asset is not None else []:
        f = el.get("file")
        if f:
            base = texdir if el.tag == "texture" else meshdir
            p = (base / f).resolve()
            if not str(p).startswith(str(asset_dir.resolve())):
                raise AssetError(f"{el.tag} file {f!r} is outside the asset's directory")
            if not p.exists():
                raise AssetError(f"{el.tag} file {f!r} does not exist")
            el.set("file", str(p))


def parse(asset_dir: Path) -> tuple[ET.Element, dict]:
    """The object's model.xml, checked against the contract's structure; its file paths made absolute."""
    path = Path(asset_dir) / "model.xml"
    if not path.exists():
        raise AssetError("no model.xml")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise AssetError(f"model.xml is not XML: {e}") from None
    if root.tag != "mujoco":
        raise AssetError("model.xml's root element is not <mujoco>")
    extra = sorted({el.tag for el in root} - ALLOWED)
    if extra:
        raise AssetError(f"model.xml may hold only {sorted(ALLOWED)} (found {extra})")
    for tag in ("include", "plugin", "keyframe", "freejoint"):
        if root.find(f".//{tag}") is not None:
            raise AssetError(f"model.xml may not use <{tag}>")
    wb = root.find("worldbody")
    bodies = [] if wb is None else [el for el in wb if el.tag == "body"]
    if len(bodies) != 1 or bodies[0].get("name") != "object" or len(wb) != 1:
        raise AssetError("worldbody must hold exactly one body, named 'object', and nothing else")
    obj = bodies[0]
    if obj.find("joint") is not None:
        raise AssetError("the 'object' body may not have a joint (the lab adds a free joint)")
    if any(obj.get(k) for k in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")):
        raise AssetError("the 'object' body must sit at the origin, unrotated")
    comp = root.find("compiler")
    if comp is None or comp.get("angle") != "radian":
        raise AssetError('model.xml needs <compiler angle="radian"/> (angles are radians, as in RoboCasa)')
    _resolve_files(root, Path(asset_dir))
    return root, {"object": obj}


def compose(asset_dir: Path, *, fixed: bool, pos=(0.0, 0.0, 1.2), R=None, extra: str = "", gravity: bool = True,
            counter: bool = True, grid: bool = False) -> str:
    """The scene XML: the object at `pos` with orientation R (object frame -> world), welded there (fixed) or free."""
    root, parts = parse(asset_dir)
    root = copy.deepcopy(root)
    obj = root.find("worldbody/body")
    obj.set("pos", " ".join(f"{v:.6f}" for v in pos))
    obj.set("quat", " ".join(f"{v:.8f}" for v in _quat(np.eye(3) if R is None else R)))
    if not fixed:
        obj.insert(0, ET.Element("freejoint", {"name": "lab_free"}))
    # The object's top-level defaults become a class of its own, so they never touch the scene's geoms.
    defaults = root.find("default")
    default_xml = ""
    if defaults is not None:
        defaults.tag = "default"
        defaults.set("class", "lab_object")
        default_xml = ET.tostring(defaults, encoding="unicode")
        obj.set("childclass", "lab_object")
    sections = {tag: "".join(ET.tostring(el, encoding="unicode") for el in root.findall(tag)) for tag in ("equality", "contact", "tendon")}
    asset_xml = "".join(ET.tostring(el, encoding="unicode") for a in root.findall("asset") for el in a)
    top = COUNTER_TOP
    counter_xml = ""
    if counter:
        counter_xml = f"""
    <geom name="lab_floor" type="plane" size="4 4 0.1" material="lab_floor"/>
    <geom name="lab_counter" type="box" pos="0 0 {top / 2}" size="0.6 0.45 {top / 2}" material="lab_counter" friction="0.6 0.005 0.0001"/>"""
        if grid:  # 5 cm squares, drawn only
            counter_xml += f"""
    <geom name="lab_grid" type="plane" pos="0 0 {top + 0.0003}" size="0.6 0.45 0.01" material="lab_grid" contype="0" conaffinity="0"/>"""
    return f"""<mujoco model="lab_scene">
  <compiler angle="radian" autolimits="true"/>
  {OPTIONS.replace('<option ', f'<option gravity="0 0 {-9.81 if gravity else 0}" ')}
  {VISUAL}
  <default>{default_xml}</default>
  <asset>{ASSETS}{asset_xml}</asset>
  <worldbody>
    <light name="lab_key" pos="0.6 -0.8 2.6" dir="-0.25 0.35 -1" diffuse="0.75 0.75 0.72" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="lab_fill" pos="-1.2 0.6 2.2" dir="0.5 -0.25 -1" diffuse="0.3 0.3 0.32" specular="0 0 0" castshadow="false"/>{counter_xml}
    {ET.tostring(obj, encoding="unicode")}
    {extra}
  </worldbody>
  {sections['equality']}{sections['contact']}{sections['tendon']}
</mujoco>"""


def load(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    except ValueError as e:
        raise AssetError(f"MuJoCo could not compile it: {e}") from None
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def subtree(m: mujoco.MjModel, root: int) -> set[int]:
    """Body ids of `root` and everything below it."""
    out = {root}
    for b in range(root + 1, m.nbody):
        if int(m.body_parentid[b]) in out:
            out.add(b)
    return out


def collision_geoms(m: mujoco.MjModel, bodies: set[int]) -> list[int]:
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]


def world_box(m: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds of these geoms (from their local boxes)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for g in geoms:
        c, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        pts = d.geom_xpos[g] + (c + signs * half) @ d.geom_xmat[g].reshape(3, 3).T
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    return lo, hi


def natural_frame(asset_dir: Path) -> Frame:
    """The object's arm, opening and hinge axes, from the object at rest (unrotated, at the origin, no gravity)."""
    m, d = load(compose(asset_dir, fixed=True, pos=(0, 0, 0), gravity=False, counter=False))
    site = lambda s: d.site_xpos[m.site(s).id].copy()
    try:
        ta, tb, ga, gb = site("tip_a"), site("tip_b"), site("grip_a"), site("grip_b")
    except KeyError as e:
        raise AssetError(f"missing site {e}") from None
    n = tb - ta
    if np.linalg.norm(n) < 1e-4:
        raise AssetError("tip_a and tip_b coincide at rest (the tongs should rest open)")
    n /= np.linalg.norm(n)
    u = (ta + tb) / 2 - (ga + gb) / 2  # from the handles toward the tips
    u -= n * (u @ n)
    if np.linalg.norm(u) < 1e-3:
        raise AssetError("the tips are not beyond the grips along the arms")
    u /= np.linalg.norm(u)
    h = np.cross(u, n)
    return Frame(u, n, h)
