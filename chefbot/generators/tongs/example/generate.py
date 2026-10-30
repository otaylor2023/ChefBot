"""Example tongs generator: a working starting point, not a realistic one. Two flat steel strips on hinges with a
spring, flat silicone pads at the tips, a cylinder for the hinge. It follows the contract (docs/generator_contract.md):
the parts, names, sites and parameters the lab needs. Everything about its shape is crude.

Frame: arms along +x (tips at +x), opening along y (arm_a at -y, arm_b at +y), hinge axis z. The object's origin is
the centre of the bounding box at rest, so the hinge sits at x = -length/2.
"""

import json
import math

import numpy as np

STEEL_DENSITY = 7900.0  # kg/m^3
SILICONE_DENSITY = 1100.0


def generate(seed: int, out_dir: str) -> dict:
    rng = np.random.default_rng(seed)
    L = float(rng.uniform(0.23, 0.36))  # hinge to tips
    gap = float(rng.uniform(0.05, 0.09))  # tip opening at rest
    w, t = 0.018, 0.0012  # strip width (z) and thickness
    pad_len, pad_t = 0.04, 0.003
    grip_x = 0.07  # where a hand squeezes, from the hinge
    k = float(rng.uniform(0.8, 2.5))  # hinge stiffness, N m / rad (each arm)
    preload = 0.25  # rad: the spring's rest angle is this far past the open stop
    half = math.atan2(gap / 2 + pad_t, L - pad_len / 2)  # each arm's angle from the centre line at rest
    x0 = -L / 2  # the hinge, so that the origin is the bounding box centre

    strip_mass = L * w * t * STEEL_DENSITY
    pad_mass = pad_len * w * pad_t * SILICONE_DENSITY
    cap_mass = 0.015

    def arm(name: str, side: int) -> str:
        """side -1: arm_a (at -y), +1: arm_b (at +y). The arm's own frame: x along it, its inner face toward y=0."""
        yaw = side * half
        q = f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}"
        inner = -side  # the inner face's y sign, in the arm's frame
        pad_y = inner * (t / 2 + pad_t / 2)
        axis = "0 0 1" if side < 0 else "0 0 -1"  # positive angle closes the arm toward the centre line
        return f"""
      <body name="{name}" pos="{x0} 0 0" quat="{q}">
        <joint name="hinge_{name[-1]}" type="hinge" axis="{axis}" stiffness="{k}" springref="{-preload}" range="0 {half + 0.1}"
               solreflimit="0.002 1" damping="0.01" armature="0.0001"/>
        <geom name="{name}_strip" type="box" pos="{L / 2} 0 0" size="{L / 2} {t / 2} {w / 2}" mass="{strip_mass}"
              friction="0.4 0.005 0.0001" rgba="0.72 0.73 0.75 1" group="0"/>
        <geom name="{name}_pad" type="box" pos="{L - pad_len / 2} {pad_y} 0" size="{pad_len / 2} {pad_t / 2} {w / 2}"
              mass="{pad_mass}" friction="1.0 0.005 0.0001" solref="0.004 1" solimp="0.9 0.95 0.002" rgba="0.8 0.15 0.12 1" group="0"/>
        <geom type="box" pos="{L / 2} 0 0" size="{L / 2} {t / 2} {w / 2}" rgba="0.72 0.73 0.75 1" contype="0" conaffinity="0" group="1" mass="0"/>
        <geom type="box" pos="{L - pad_len / 2} {pad_y} 0" size="{pad_len / 2} {pad_t / 2} {w / 2}" rgba="0.8 0.15 0.12 1"
              contype="0" conaffinity="0" group="1" mass="0"/>
        <site name="grip_{name[-1]}" pos="{grip_x} {-inner * t / 2} 0" size="0.004"/>
        <site name="tip_{name[-1]}" pos="{L - pad_len / 2} {inner * (t / 2 + pad_t)} 0" size="0.004"/>
      </body>"""

    xml = f"""<mujoco model="tongs_{seed}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="object">
      <geom name="hinge_cap" type="cylinder" pos="{x0} 0 0" size="0.008 {w / 2 + 0.002}" mass="{cap_mass}" rgba="0.6 0.6 0.62 1"
            friction="0.4 0.005 0.0001" group="0"/>
      <geom type="cylinder" pos="{x0} 0 0" size="0.008 {w / 2 + 0.002}" rgba="0.6 0.6 0.62 1" contype="0" conaffinity="0" group="1" mass="0"/>{arm("arm_a", -1)}{arm("arm_b", +1)}
    </body>
  </worldbody>
  <equality>
    <joint joint1="hinge_a" joint2="hinge_b" polycoef="0 1 0 0 0" solref="0.005 1"/>
  </equality>
</mujoco>
"""
    with open(f"{out_dir}/model.xml", "w") as f:
        f.write(xml)

    # The tips touch when each arm has turned by about `half` less the pad's share; each arm's spring then pushes
    # back with k * (q + preload), against the force F at grip_x: F * grip_x = k * (q + preload).
    q_touch = half - math.atan2(pad_t + t / 2, L - pad_len / 2)
    closing_force = k * (q_touch + preload) / grip_x
    params = {
        "length_m": round(L + 0.008, 4),  # plus the hinge cap
        "mass_kg": round(2 * (strip_mass + pad_mass) + cap_mass, 4),
        "rest_tip_gap_m": round(gap, 4),
        "closing_force_N": round(closing_force, 2),
        "tip_friction": 1.0,
        "style": "flat strips, flat silicone pads, no lock",
        "spring_stiffness_Nm_per_rad": round(k, 3),
    }
    print(json.dumps(params))
    return params
