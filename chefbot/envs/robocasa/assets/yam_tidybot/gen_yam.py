"""Generate robosuite robot.xml (TidyBot++ chassis + YAM arm) and gripper.xml (linear 4310 + D405)
from i2rt's yam_station_linear_4310_d405.xml (left arm chain)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from chefbot.envs.robocasa.yam_tidybot import MOUNT_X, MOUNT_Z, ZED_IN_ARM, zed_in_support  # noqa: E402

STATION = Path.home() / "dev/i2rt/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml"
OUT = Path(sys.argv[1])
PLATE_Z = 0.335  # top of the TidyBot++ arm plate
CENTER_X = -0.21  # NullMobileBase's yaw axis, in its support frame
TORQUE = [28, 28, 28, 10, 10, 10]  # DM4340 x3, DM4310 x3 (i2rt motor TORQUE_MAX)
DAMPING = [0.5, 0.5, 0.5, 0.2, 0.1, 0.1]
ARMATURE = [0.03, 0.03, 0.03, 0.01, 0.01, 0.01]

root = ET.parse(STATION).getroot()
body = {b.get("name"): b for b in root.iter("body")}


def q_of(el):
    return np.array([float(x) for x in (el.get("quat") or "1 0 0 0").split()])


def p_of(el):
    return np.array([float(x) for x in (el.get("pos") or "0 0 0").split()])


def R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def fmt(v):
    return " ".join(f"{x:.7g}" for x in v)


def wxyz(Rm):
    x, y, z, w = Rotation.from_matrix(Rm).as_quat()
    return [w, x, y, z]


# ---------------- arm ----------------
def strip(el, keep_children=("body",)):
    """Copy a body without its children bodies (we rebuild the chain)."""
    out = ET.Element("body", {k: v for k, v in el.attrib.items()})
    for c in el:
        if c.tag != "body":
            out.append(c)
    return out


chain = ["left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_gripper"]
names = ["link1", "link2", "link3", "link4", "link5", "link6"]
arm_root = ET.Element("body", {"name": "link0", "pos": f"{MOUNT_X} 0 {MOUNT_Z}"})
ET.SubElement(arm_root, "site", {"name": "right_center", "pos": "0 0 0", "size": "0.01", "rgba": "1 0.3 0.3 0", "group": "2"})
base_geom = body["left_base"].find("geom")
for g in ("vis", "col"):
    a = {"pos": base_geom.get("pos"), "quat": base_geom.get("quat"), "type": "mesh", "mesh": "yam_base"}
    if g == "vis":
        a.update({"rgba": "0.2 0.2 0.22 1", "contype": "0", "conaffinity": "0", "group": "1", "name": "link0_vis"})
    else:
        a.update({"group": "0", "name": "link0_col"})
    ET.SubElement(arm_root, "geom", a)
inert = body["left_base"].find("inertial")
arm_root.insert(0, ET.Element("inertial", inert.attrib))
parent = arm_root
for i, (src, dst) in enumerate(zip(chain, names)):
    b = body[src]
    nb = ET.SubElement(parent, "body", {"name": dst, "pos": b.get("pos"), "quat": b.get("quat")})
    nb.append(ET.Element("inertial", b.find("inertial").attrib))
    j = b.find("joint")
    ET.SubElement(nb, "joint", {"name": f"joint{i + 1}", "pos": "0 0 0", "axis": j.get("axis"), "type": "hinge", "limited": "true",
                                "range": j.get("range"), "damping": str(DAMPING[i]), "armature": str(ARMATURE[i]), "frictionloss": "0.1"})
    if dst != "link6":
        g = b.find("geom")
        mesh = g.get("mesh")
        for kind in ("vis", "col"):
            a = {"pos": g.get("pos"), "quat": g.get("quat"), "type": "mesh", "mesh": mesh, "name": f"{dst}_{kind}"}
            if kind == "vis":
                a.update({"rgba": "0.82 0.82 0.84 1" if i % 2 else "0.2 0.2 0.22 1", "contype": "0", "conaffinity": "0", "group": "1"})
            else:
                a["group"] = "0"
            ET.SubElement(nb, "geom", a)
    else:
        # The gripper (mesh, fingers, camera) is the robosuite gripper model, attached at right_hand
        # (= the flange: +Z approach, +X across the fingers, +Y down at home).
        nb.find("inertial").attrib.update({"pos": "0 0 0", "mass": "0.01", "diaginertia": "1e-5 1e-5 1e-5"})
        nb.find("inertial").attrib.pop("quat", None)
        rh = ET.SubElement(nb, "body", {"name": "right_hand", "pos": "0 0 0", "quat": "1 0 0 0"})
        ET.SubElement(rh, "inertial", {"pos": "0 0 0", "mass": "1e-4", "diaginertia": "1e-7 1e-7 1e-7"})
        ET.SubElement(rh, "site", {"name": "flange", "pos": "0 0 0", "size": "0.005", "rgba": "0 0 1 0", "group": "2"})
    parent = nb

mj = ET.Element("mujoco", {"model": "yam_tidybot"})
ET.SubElement(mj, "compiler", {"angle": "radian"})
act = ET.SubElement(mj, "actuator")
for i, t in enumerate(TORQUE):
    ET.SubElement(act, "motor", {"ctrllimited": "true", "ctrlrange": f"{-t} {t}", "joint": f"joint{i + 1}", "name": f"torq_j{i + 1}"})
asset = ET.SubElement(mj, "asset")
for n, c in (("tb_black", "0.11 0.11 0.10 1"), ("tb_gray", "0.63 0.65 0.65 1"), ("tb_dark", "0.35 0.38 0.37 1")):
    ET.SubElement(asset, "material", {"name": n, "rgba": c})
for n in ("bumper", "bottom_plate", "body", "top_plate", "arm_plate"):
    ET.SubElement(asset, "mesh", {"name": f"tb_{n}", "file": f"meshes/base/{n}.stl"})
ET.SubElement(asset, "mesh", {"name": "yam_base", "file": "meshes/yam/base.stl"})
for n in ("link1", "link2", "link3", "link4", "link5"):
    ET.SubElement(asset, "mesh", {"name": n, "file": f"meshes/yam/{n}.stl"})
wb = ET.SubElement(mj, "worldbody")
base0 = ET.SubElement(wb, "body", {"name": "base", "pos": "0 0 0"})
ET.SubElement(base0, "inertial", {"pos": f"{CENTER_X} 0 0.15", "mass": "20", "diaginertia": "0.5 0.5 0.5"})
# Third-person view for videos: behind and to the left of the robot, high, looking at the arm.
_eye, _at = np.array([CENTER_X - 1.1, 0.9, 1.7]), np.array([CENTER_X + 0.35, 0.0, 0.8])
_f = (_at - _eye) / np.linalg.norm(_at - _eye)
_r = np.cross(_f, [0.0, 0.0, 1.0]); _r /= np.linalg.norm(_r)
ET.SubElement(base0, "camera", {"mode": "fixed", "name": "robotview", "pos": fmt(_eye), "quat": fmt(wxyz(np.column_stack([_r, np.cross(_r, _f), -_f]))), "fovy": "60"})
# robosuite's NullMobileBase turns about a point 0.21 m behind its origin: center the robot there.
base = ET.SubElement(base0, "body", {"name": "chassis", "pos": f"{CENTER_X} 0 0"})
# robosuite drops the root body's inertial when it adds the mobile base: the chassis carries the mass
# (TidyBot++ is about 25 kg; a 0.48 m square box).
ET.SubElement(base, "inertial", {"pos": "0 0 0.15", "mass": "25", "diaginertia": "0.67 0.67 0.96"})
for n, mat in (("bumper", "tb_black"), ("bottom_plate", "tb_dark"), ("body", "tb_gray"), ("top_plate", "tb_dark"), ("arm_plate", "tb_gray")):
    ET.SubElement(base, "geom", {"name": f"{n}_vis", "type": "mesh", "mesh": f"tb_{n}", "material": mat, "contype": "0", "conaffinity": "0", "group": "1"})
# Chassis collision: one box (the bumper's footprint, floor clearance 27 mm) and the arm plate.
ET.SubElement(base, "geom", {"name": "chassis_col", "type": "box", "pos": "0 0 0.181", "size": "0.241 0.241 0.154", "group": "0", "rgba": "0.5 0.5 0.5 0"})
ET.SubElement(base, "geom", {"name": "arm_plate_col", "type": "box", "pos": "0 0 0.33", "size": "0.075 0.175 0.005", "group": "0", "rgba": "0.5 0.5 0.5 0"})
# The YAM's riser (plate to MOUNT_Z), the pole for the high ZED, and the two ZEDs' housings (behind
# each camera's optical center, so they never show in its own image).
_rz = (MOUNT_Z - PLATE_Z) / 2
for kind in ("vis", "col"):
    a = {"name": f"riser_{kind}", "type": "box", "pos": fmt([MOUNT_X, 0, PLATE_Z + _rz]), "size": fmt([0.06, 0.06, _rz])}
    a.update({"material": "tb_dark", "contype": "0", "conaffinity": "0", "group": "1"} if kind == "vis" else {"group": "0", "rgba": "0.5 0.5 0.5 0"})
    ET.SubElement(base, "geom", a)
for name in ZED_IN_ARM:
    pos, Rm = zed_in_support(name)
    pos = pos - np.array([CENTER_X, 0, 0])  # chassis frame
    behind = pos + Rm[:, 2] * 0.03  # MuJoCo camera looks along -z
    ET.SubElement(base, "geom", {"name": f"{name[7:]}_housing_vis", "type": "box", "pos": fmt(behind), "quat": fmt(wxyz(Rm)),
                                 "size": "0.0875 0.015 0.0165", "material": "tb_black", "contype": "0", "conaffinity": "0", "group": "1"})
    if "pole" in name:
        top = behind - np.array([0, 0, 0.02])
        ET.SubElement(base, "geom", {"name": "pole_vis", "type": "cylinder", "fromto": fmt([*top[:2], PLATE_Z, *top]), "size": "0.012",
                                     "material": "tb_black", "contype": "0", "conaffinity": "0", "group": "1"})
base.append(arm_root)
ET.indent(mj)
(OUT / "robot.xml").write_text("<!-- Generated by gen_yam.py: TidyBot++ chassis (stanford-iprl-lab/tidybot_robosuite) with an I2RT YAM v1\n"
                                "     (i2rt yam_station_linear_4310_d405.xml, left arm) on a riser, and MessyMem's ZED pole. Torque limits: i2rt TORQUE_MAX. -->\n"
                                + ET.tostring(mj, encoding="unicode"))

# ---------------- gripper ----------------
G = body["left_gripper"]
gg = G.find("geom")
gm = ET.Element("mujoco", {"model": "yam_linear_4310"})
ET.SubElement(gm, "compiler", {"angle": "radian"})
ga = ET.SubElement(gm, "asset")
for n in ("gripper", "tip_left", "tip_right", "d405_wrist_linear_4310"):
    ET.SubElement(ga, "mesh", {"name": n, "file": f"meshes/gripper/{n}.stl"})
ET.SubElement(ga, "material", {"name": "g_dark", "rgba": "0.15 0.15 0.16 1"})
ET.SubElement(ga, "material", {"name": "g_tip", "rgba": "0.6 0.6 0.62 1"})
gwb = ET.SubElement(gm, "worldbody")
hb = ET.SubElement(gwb, "body", {"name": "hand", "pos": "0 0 0"})
hb.append(ET.Element("inertial", G.find("inertial").attrib))
ET.SubElement(hb, "site", {"name": "ft_frame", "pos": "0 0 0", "size": "0.01", "rgba": "1 0 0 0", "type": "sphere", "group": "1"})
for kind in ("visual", "collision"):
    a = {"name": f"hand_{kind}", "type": "mesh", "mesh": "gripper", "pos": gg.get("pos"), "quat": gg.get("quat")}
    a.update({"material": "g_dark", "contype": "0", "conaffinity": "0", "group": "1"} if kind == "visual" else {"group": "0"})
    ET.SubElement(hb, "geom", a)
# Grip site: between the fingertip faces (i2rt's grasp_site), z along the approach, x along the closing direction.
eef = ET.SubElement(hb, "body", {"name": "eef", "pos": "0 0 0.1447"})
for n, a in (("grip_site", {"size": "0.01", "type": "sphere"}),
             ("ee_x", {"pos": "0.1 0 0", "size": "0.005 .1", "quat": "0.707105 0 0.707108 0", "type": "cylinder"}),
             ("ee_y", {"pos": "0 0.1 0", "size": "0.005 .1", "quat": "0.707105 0.707108 0 0", "type": "cylinder"}),
             ("ee_z", {"pos": "0 0 0.1", "size": "0.005 .1", "type": "cylinder"}),
             ("grip_site_cylinder", {"size": "0.005 10", "type": "cylinder"})):
    ET.SubElement(eef, "site", {"name": n, "rgba": "1 0 0 0", "group": "1", **a})
# Wrist D405 (i2rt's mount); its optical frame is OpenCV (+z forward, +y down): MuJoCo cameras look down -z with +y up.
cam = body["left_camera"]
cb = ET.SubElement(hb, "body", {"name": "d405", "pos": cam.get("pos"), "quat": cam.get("quat")})
ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": "0.03", "diaginertia": "1e-6 1e-6 1e-6"})
ET.SubElement(cb, "geom", {"name": "d405_visual", "type": "mesh", "mesh": "d405_wrist_linear_4310", "material": "g_dark", "contype": "0", "conaffinity": "0", "group": "1"})
ET.SubElement(cb, "camera", {"name": "wrist_d405", "mode": "fixed", "pos": "0 0 0", "quat": "0 1 0 0", "fovy": "58"})
# Fingers: tip pads are boxes on the inner faces (27 mm along the approach, 28 mm wide), the stems a box each.
for side, sign in (("left", -1), ("right", 1)):
    tb = body[f"left_tip_{side}"]
    tq, tp = q_of(tb), p_of(tb)
    Rt = R(tq)
    nb = ET.SubElement(hb, "body", {"name": f"{side}_finger", "pos": fmt(tp), "quat": fmt(tq)})
    nb.append(ET.Element("inertial", tb.find("inertial").attrib))
    j = tb.find("joint")
    ET.SubElement(nb, "joint", {"name": f"{side}_finger_joint", "type": "slide", "axis": j.get("axis"), "limited": "true", "range": "0 0.0469",
                                "damping": "2", "armature": "0.01", "frictionloss": "0.5"})
    tg = tb.find("geom")
    ET.SubElement(nb, "geom", {"name": f"{side}_tip_visual", "type": "mesh", "mesh": f"tip_{side}", "pos": tg.get("pos"), "quat": tg.get("quat"),
                               "material": "g_tip", "contype": "0", "conaffinity": "0", "group": "1"})

    def box(name, lo, hi, extra):
        # lo/hi: flange-frame corners at q=0 -> box in the finger frame
        c = (np.array(lo) + np.array(hi)) / 2
        h = (np.array(hi) - np.array(lo)) / 2
        cl = Rt.T @ (c - tp)
        hl = np.abs(Rt.T @ h)
        ET.SubElement(nb, "geom", {"name": name, "type": "box", "pos": fmt(cl), "size": fmt(hl), "group": "0", "rgba": "0.3 0.3 0.3 1", **extra})

    # At q=0 (closed) the inner faces meet at x=0; right finger sits at +x.
    pad_x = (0.0, 0.0064) if sign > 0 else (-0.0064, 0.0)
    box(f"{side}_pad_collision", (pad_x[0], -0.014, 0.12), (pad_x[1], 0.014, 0.147),
        {"friction": "1.0 0.01 0.001", "solimp": "0.95 0.99 0.001", "solref": "0.004 1", "priority": "1"})
    stem_x = (0.0025, 0.023) if sign > 0 else (-0.023, -0.0025)
    box(f"{side}_stem_collision", (stem_x[0], -0.014, 0.08), (stem_x[1], 0.014, 0.12), {})
ET.SubElement(gm, "contact")
for a, b in (("hand", "left_finger"), ("hand", "right_finger"), ("left_finger", "right_finger")):
    ET.SubElement(gm.find("contact"), "exclude", {"body1": a, "body2": b})
eq = ET.SubElement(gm, "equality")
ET.SubElement(eq, "joint", {"name": "finger_coupling", "joint1": "left_finger_joint", "joint2": "right_finger_joint", "polycoef": "0 1 0 0 0",
                            "solimp": "0.95 0.99 0.001", "solref": "0.005 1"})
ga2 = ET.SubElement(gm, "actuator")
# Position servo on the finger opening (0 closed .. 0.0469 open); the grip force is capped at 50 N (i2rt's limit).
ET.SubElement(ga2, "position", {"name": "fingers_actuator", "joint": "left_finger_joint", "kp": "4000", "kv": "35", "ctrllimited": "true",
                                "ctrlrange": "0 0.0469", "forcelimited": "true", "forcerange": "-50 50"})
sen = ET.SubElement(gm, "sensor")
ET.SubElement(sen, "force", {"name": "force_ee", "site": "ft_frame"})
ET.SubElement(sen, "torque", {"name": "torque_ee", "site": "ft_frame"})
ET.indent(gm)
(OUT / "gripper.xml").write_text("<!-- Generated by gen_yam.py: I2RT linear_4310 gripper with its wrist D405, in the YAM flange frame\n"
                                  "     (+Z approach, +X across the fingers). Geometry from i2rt yam_station_linear_4310_d405.xml. -->\n"
                                  + ET.tostring(gm, encoding="unicode"))
print("ok")

# ---------------- kinematics (IK) ----------------
# The arm alone, base frame at the origin (= the arm mount), with the flange as `attachment_site`.
km = ET.Element("mujoco", {"model": "yam_kinematics"})
ET.SubElement(km, "compiler", {"angle": "radian"})
kwb = ET.SubElement(km, "worldbody")
kparent = ET.SubElement(kwb, "body", {"name": "yam_link0", "pos": "0 0 0"})
ET.SubElement(kparent, "inertial", {"pos": "0 0 0", "mass": "1", "diaginertia": "0.01 0.01 0.01"})
for i, (src, dst) in enumerate(zip(chain, names)):
    b = body[src]
    nb = ET.SubElement(kparent, "body", {"name": f"yam_{dst}", "pos": b.get("pos"), "quat": b.get("quat")})
    ET.SubElement(nb, "inertial", {"pos": "0 0 0", "mass": "0.5", "diaginertia": "0.001 0.001 0.001"})
    j = b.find("joint")
    ET.SubElement(nb, "joint", {"name": f"yam_joint{i + 1}", "axis": j.get("axis"), "type": "hinge", "limited": "true", "range": j.get("range")})
    kparent = nb
ET.SubElement(kparent, "site", {"name": "attachment_site", "pos": "0 0 0", "size": "0.005"})
ET.indent(km)
kin = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if kin:
    kin.write_text("<!-- Generated by gen_yam.py: YAM v1 kinematics (i2rt yam_station_linear_4310_d405.xml, left arm) for IK.\n"
                   "     Arm base frame at the origin; attachment_site is the flange (+Z approach, +X across the fingers). -->\n"
                   + ET.tostring(km, encoding="unicode"))
    print("kinematics ok")
