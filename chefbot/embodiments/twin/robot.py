"""The robot's digital twin: the TidyBot++ with the FR3 and Franka Hand, assembled from MuJoCo Menagerie (stanford_tidybot, franka_fr3,
franka_emika_panda) as an MjSpec to attach to a scene.

Base: Menagerie's holonomic base on x / y slides and a yaw hinge (world frame), position actuators; the arm on its
arm plate, 0.335 m up. Hand: on the FR3's flange, turned -45 degrees about it as on the real arm (hand.xml's body adds
180). The hand's actuator is a servo on the finger tendon (the fingers' mean opening, m): ctrl is its target, and
hand.py's FrankaHand drives it (and its force limit) as libfranka's move and grasp. The pads get torsional friction
(condim 6): a held tool should not spin in them.

Cameras: `wrist` (beside the hand, looking along the fingers) and `pole` (on a mast at the back of the base, looking
forward and down), both RealSense-like.

    spec = robot_spec()                                  # base_link at the origin, names prefixed by the caller
    spec = robot_spec(fingers="fingers_run2_final")      # wearing printed fingers (embodiments/grippers/<name>)
    MENAGERIE: $MUJOCO_MENAGERIE or ~/dev/mujoco_menagerie
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path(os.environ.get("MUJOCO_MENAGERIE", Path.home() / "dev" / "mujoco_menagerie"))
ARM_MOUNT_Z = 0.335  # the arm plate's top, above the floor
HAND_QUAT = [0.3826834, 0.0, 0.0, 0.9238795]  # on the flange: +45 deg about z; hand.xml's body adds 180
GRIP_KP = 3000.0  # N per m of finger travel (the tendon is the fingers' mean)
GRIP_KV = 60.0
GRIP_KP_MOVE, GRIP_KV_MOVE = 10000.0, 60.0  # a move's servo: stiffer (a spindle drive; light resistance hardly moves it)
FINGER_TRAVEL = 0.04  # each finger, open to closed
PAD_FRICTION = [1.0, 0.02, 0.002]
WRIST_CAM = {"pos": [0.0, -0.065, 0.02], "pitch_deg": 12.0, "fovy": 58.0}  # in the hand frame
POLE_CAM = {"pos": [-0.22, 0.0, 1.55], "look_at": [0.9, 0.0, 0.85], "fovy": 50.0}  # in the base frame


def _absolute_files(spec: mujoco.MjSpec, base: Path) -> None:
    for item in [*spec.meshes, *spec.textures]:
        if item.file and not Path(item.file).is_absolute():
            item.file = str((base / item.file).resolve())
    spec.meshdir = ""
    spec.texturedir = ""


def _look_quat(pos, target) -> list[float]:
    """A MuJoCo camera quat (it looks along its -z, +y up) at `pos` looking at `target`, level."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q.tolist()


GRIPPERS = Path(__file__).resolve().parents[1] / "grippers"
# A printed finger from the gripper lab, in its finger frame (mm: origin at the carriage post's mounting face, x toward
# the other finger, y along the post, z down the finger) -> the Franka Hand's finger body (m: origin on the hand axis
# 58.4 mm below the hand, +y opening, +z approach). x_m = y_g, y_m = 15.5 mm - x_g, z_m = z_g - 5.5 mm: the mount face
# is 15.5 mm off the axis at width 0 (the stock pads' faces meet on it), and the post's centre 5.5 mm above the body's
# origin (the stock pads' centre, 50 mm below the post's, is at 44.5 mm).
FINGER_FRAME_POS = [0.0, 0.0155, -0.0055]
FINGER_FRAME_QUAT = [0.7071068, 0.0, 0.0, -0.7071068]  # x_g -> -y_m, y_g -> +x_m, z_g -> +z_m
PRINTED_PAD_CENTER_M = 0.091  # fingers_run2_final's pads' middle out from the flange (stock: 0.1034)


def sim_fingers(name: str | None = None) -> Path | None:
    """The printed fingers the hand wears: `name` (a folder in embodiments/grippers), else $CHEFBOT_SIM_FINGERS, else
    the stock Franka fingers (None), which are what the real robot wears."""
    name = name or os.environ.get("CHEFBOT_SIM_FINGERS")
    return None if not name or name == "stock" else GRIPPERS / name


def _swap_fingers(hand: mujoco.MjSpec, asset: Path) -> None:
    """Both fingers' stock geoms out, the printed parts (seen) and their collision pieces (felt, each with its
    material's friction and softness) in; the finger bodies, joints, coupling and actuator stay."""
    import json

    import yaml

    parts = json.loads((asset / "parts.json").read_text())["parts"]
    pieces = json.loads((asset / "collision.json").read_text())["pieces"]
    mats = yaml.safe_load((asset / "materials.yaml").read_text())
    for k, p in enumerate(parts):
        hand.add_mesh(name=f"pf_part{k}", file=str((asset / p["file"]).resolve()), scale=[0.001] * 3)
    for k, c in enumerate(pieces):
        hand.add_mesh(name=f"pf_col{k}", file=str((asset / c["file"]).resolve()), scale=[0.001] * 3)
    for side in ("left_finger", "right_finger"):
        body = hand.body(side)
        for g in list(body.geoms):  # the stock finger and its pads: switched off (no contact, not drawn, no mass)
            g.contype, g.conaffinity, g.group, g.mass = 0, 0, 5, 0.0
        for k, p in enumerate(parts):
            soft = p.get("role") != "body"
            rgba = [0.95, 0.55, 0.15, 1] if soft else [0.35, 0.45, 0.6, 1]
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"pf_part{k}", contype=0, conaffinity=0, group=2,
                          mass=0, rgba=rgba, pos=FINGER_FRAME_POS, quat=FINGER_FRAME_QUAT)  # fmt: skip
        for k, c in enumerate(pieces):
            m = mats[c["material"]]
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"pf_col{k}", group=3, mass=0.0006,
                              condim=int(m["condim"]), friction=list(m["friction"]), priority=1, rgba=[0, 0, 0, 0],
                              pos=FINGER_FRAME_POS, quat=FINGER_FRAME_QUAT)  # fmt: skip
            g.solref = [float(m["solref"]), 1.0]


def arm_spec(fingers: str | None = None) -> mujoco.MjSpec:
    """The FR3 with the Franka Hand on its flange, ChefBot's gripper servo and pads."""
    arm = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_fr3" / "fr3.xml"))
    _absolute_files(arm, MENAGERIE / "franka_fr3" / "assets")
    hand = mujoco.MjSpec.from_file(str(MENAGERIE / "franka_emika_panda" / "hand.xml"))
    _absolute_files(hand, MENAGERIE / "franka_emika_panda" / "assets")
    act = hand.actuator("actuator8")
    # ctrl: the tendon's target (each finger's opening, m). The env sets the force range per command (on the tendon,
    # twice the fingers' squeeze: each finger gets half) and, while a grasp squeezes, makes it a plain force.
    act.gainprm[0] = GRIP_KP
    act.biasprm[:3] = [0.0, -GRIP_KP, -GRIP_KV]
    act.ctrlrange = [-1000.0, FINGER_TRAVEL]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    for g in hand.geoms:
        if g.classname.name.startswith("fingertip_pad_collision"):
            g.condim = 6
            g.friction = PAD_FRICTION
            g.priority = 1
    fingers = sim_fingers(fingers)
    pinch = 0.1034  # Franka's documented flange -> TCP, the stock pads' centre
    if fingers is not None:
        _swap_fingers(hand, fingers)
        pinch = PRINTED_PAD_CENTER_M
    pitch = math.radians(WRIST_CAM["pitch_deg"])
    # Looking along the hand's +z (cam -z = hand +z), tipped toward the fingers.
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    tilt = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (tilt @ R).ravel())
    hand.body("hand").add_camera(name="wrist", pos=WRIST_CAM["pos"], quat=q.tolist(), fovy=WRIST_CAM["fovy"])
    hand.body("hand").add_site(name="pinch", pos=[0, 0, pinch], size=[0.004, 0, 0], group=5)
    for body in [*arm.bodies, *hand.bodies]:  # the real arm's controller compensates gravity (per body: not inherited)
        body.gravcomp = 1.0
    flange = arm.site("attachment_site")
    frame = flange.parent.add_frame(pos=flange.pos, quat=HAND_QUAT)
    frame.attach_body(hand.body("hand"), "hand/", "")
    return arm


def _add_body_parts(link) -> None:
    """The real robot's fixed parts around the arm (embodiments/tidybot_fr3/robot.yaml `body:`: the pole, the
    camera on its clamp), solid, at their measured places relative to the arm (on its plate here, ARM_MOUNT_Z up):
    the controller steers clear of them, and the arm stops on them if it doesn't. (The thin mast above, the sim's
    pole camera's, is only seen.)"""
    from chefbot.interface.workspace import load_body

    body = load_body("tidybot_fr3")
    up = np.array([0.0, 0.0, ARM_MOUNT_Z])
    for name, xy, r, top in body.cylinders:
        link.add_geom(name=name.replace(" ", "_"), type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, top / 2, 0],
                      pos=[xy[0], xy[1], ARM_MOUNT_Z + top / 2], rgba=[0.6, 0.6, 0.62, 1], mass=0)  # fmt: skip
    for name, a, b, r in body.capsules:
        link.add_geom(name=name.replace(" ", "_"), type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, 0, 0],
                      fromto=[*(np.asarray(a) + up), *(np.asarray(b) + up)], rgba=[0.2, 0.2, 0.2, 1], mass=0)  # fmt: skip


def robot_spec(fingers: str | None = None) -> mujoco.MjSpec:
    """The whole robot: base_link (with its slides and yaw) at the origin, the arm on it as fr3/, the hand as
    fr3/hand/."""
    base = mujoco.MjSpec.from_file(str(MENAGERIE / "stanford_tidybot" / "base.xml"))
    _absolute_files(base, MENAGERIE / "stanford_tidybot" / "assets")
    link = base.body("base_link")
    link.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.6, 0], pos=[POLE_CAM["pos"][0], 0, 0.95],
                  rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0, group=2, mass=0)  # fmt: skip
    link.add_camera(name="pole", pos=POLE_CAM["pos"], quat=_look_quat(POLE_CAM["pos"], POLE_CAM["look_at"]), fovy=POLE_CAM["fovy"])
    _add_body_parts(link)
    mount = link.add_frame(pos=[0, 0, ARM_MOUNT_Z])
    mount.attach_body(arm_spec(fingers).body("base"), "fr3/", "")
    return base
