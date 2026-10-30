"""A plain-MuJoCo scene: a folder with world.xml (the model), annotations.yaml (what its parts are, what the scene
senses and shows, its run parameters) and optionally dynamics.py (the scene's own logic, run sandboxed by the
environment). Generated scenes (chefbot/generators/) and exported ones share this format.

annotations.yaml:

    control_hz: 50                       # how often the dynamics run
    dynamics: dynamics.py                # class Dynamics(annotations, params, seed) with step(t, sensed) -> commands
    params: {...}                        # run parameters: the dynamics' defaults (a setup may override them)
    objects: {name: {kind: ..., ...}}    # the scene's parts by kind (kinds.py): what tasks and checks address
    sensors:                             # what the dynamics may read, by name
      <name>: {joint: <joint>}           #   position (slide m, hinge rad) and velocity
      <name>: {actuator_force: <act>}    #   the actuator's force (N or N m)
      <name>: {beam: [<site>, <site>]}   #   True when anything blocks the line between the sites
      <name>: {load: <body>}             #   kg resting on the body (its contact normal forces / g)
    actuators: [<actuator>, ...]         # the actuators the dynamics may set (their ctrl)
    displays:                            # what the dynamics may show
      <display>: {fields: {<field>: {default: <state>, states: {<state>: [[<geom>, {rgba: [...]}|{material: name}]]}}}}

A display state lists geoms and their colour or material; selecting it applies them (geoms another state of the same
field changes are reset to the model's own first).
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import yaml


class SceneError(ValueError):
    """The scene breaks the format (or does not compile)."""


def load_annotations(folder: Path) -> dict:
    p = Path(folder) / "annotations.yaml"
    if not p.exists():
        raise SceneError("no annotations.yaml")
    try:
        ann = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise SceneError(f"annotations.yaml: {e}") from None
    for key in ("objects", "sensors", "displays", "params"):
        ann.setdefault(key, {})
    ann.setdefault("actuators", [])
    ann.setdefault("control_hz", 50)
    return ann


def compose(folder: Path, extra: str = "", extra_assets: str = "", extra_actuators: str = "") -> str:
    """world.xml with `extra` (XML) added to its worldbody, `extra_assets` to its assets and `extra_actuators` to its
    actuators, file paths made absolute."""
    folder = Path(folder).resolve()
    path = folder / "world.xml"
    if not path.exists():
        raise SceneError("no world.xml")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise SceneError(f"world.xml is not XML: {e}") from None
    root = copy.deepcopy(root)
    comp = root.find("compiler")
    if comp is None or comp.get("angle") != "radian":
        raise SceneError('world.xml needs <compiler angle="radian"/>')
    for tag in ("include", "plugin"):
        if root.find(f".//{tag}") is not None:
            raise SceneError(f"world.xml may not use <{tag}>")
    meshdir = folder / comp.get("meshdir", "")
    texdir = folder / comp.get("texturedir", "")
    for k in ("meshdir", "texturedir", "assetdir"):
        comp.attrib.pop(k, None)
    for el in root.iter():
        f = el.get("file")
        if f and el.tag in ("mesh", "texture", "hfield", "skin"):
            p = ((texdir if el.tag == "texture" else meshdir) / f).resolve()
            if not str(p).startswith(str(folder)):
                raise SceneError(f"{el.tag} file {f!r} is outside the scene's folder")
            el.set("file", str(p))
    wb = root.find("worldbody")
    if wb is None:
        raise SceneError("world.xml has no worldbody")
    if extra:
        for el in ET.fromstring(f"<x>{extra}</x>"):
            wb.append(el)
    if extra_assets:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        for el in ET.fromstring(f"<x>{extra_assets}</x>"):
            asset.append(el)
    if extra_actuators:
        act = root.find("actuator")
        if act is None:
            act = ET.SubElement(root, "actuator")
        for el in ET.fromstring(f"<x>{extra_actuators}</x>"):
            act.append(el)
    return ET.tostring(root, encoding="unicode")


def load_model(folder: Path, extra: str = "", extra_assets: str = "", extra_actuators: str = "") -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_string(compose(folder, extra, extra_assets, extra_actuators))
    except ValueError as e:
        raise SceneError(f"MuJoCo could not compile it: {e}") from None
