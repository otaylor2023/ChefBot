"""Example elevator generator: a working starting point, not a realistic one. One car in one shaft, `floors` floors
(default 2), box geometry, centre-opening doors, round buttons, indicator lamps, and a simple controller
(dynamics.py) with one fault. It follows the contract (docs/generator_contract.md): the parts, names, sites,
sensors and parameters the lab needs. Everything about how it looks, and much about how it behaves, is crude.

Frame: the landings are at x < 0 (the hall), the shaft and car at x > 0, z up; floor k's landing floor is at
z = k * floor_height. The car's doorway faces -x.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOOR_W = 1.0  # clear opening
CAR_D, CAR_W, CAR_H = 1.6, 2.0, 2.4  # car depth (x), width (y), height
PANEL_T = 0.025  # door panel thickness


def _f(v) -> str:
    return " ".join(f"{x:.5f}" for x in np.atleast_1d(v))


def generate(seed: int, out_dir: str, overrides: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    o = dict(overrides or {})
    floors = int(o.get("floors", 2))
    floor_h = float(o.get("floor_height_m", rng.uniform(3.6, 4.2)))
    gap = float(o.get("sill_gap_m", rng.uniform(0.018, 0.03)))
    run = {
        "max_wait_s": 0.0,  # filled below from travel
        "door_open_s": float(rng.uniform(2.0, 3.0)),
        "door_close_s": float(rng.uniform(2.5, 3.5)),
        "dwell_s": float(rng.uniform(3.0, 5.0)),
        "rated_speed_mps": float(rng.uniform(1.0, 1.75)),
        "accel_mps2": float(rng.uniform(0.8, 1.0)),
        "jerk_mps3": float(rng.uniform(1.5, 2.0)),
        "leveling_tol_m": 0.006,
        "door_force_limit_N": 100.0,
        "fault_missed_press_rate": 0.03,
    }
    travel = (floors - 1) * floor_h
    run["max_wait_s"] = round(run["door_close_s"] + run["dwell_s"] + travel / run["rated_speed_mps"] + 2 * run["rated_speed_mps"] / run["accel_mps2"]
                              + run["accel_mps2"] / run["jerk_mps3"] * 2 + 2.0, 1)  # fmt: skip
    for k, v in o.items():
        if k in run:
            run[k] = v
    build = {"floors": floors, "cars": 1, "floor_height_m": round(floor_h, 3), "sill_gap_m": round(gap, 4), "building_type": "office",
             "door_type": "center_opening", "car_depth_m": CAR_D, "car_width_m": CAR_W, "door_width_m": DOOR_W}  # fmt: skip

    labels = ["L"] + [str(k + 1) for k in range(1, floors)]
    geoms, landing_bodies, buttons, objects, sensors, displays, actuators = [], [], [], {}, {}, {}, []
    half_door = DOOR_W / 2

    def button(bid: str, pos, normal: str, meaning: dict, parent: list):
        """A round button on a slide joint; its face points along `normal` (-x: into the hall; +y: into the car)."""
        quat = {"-x": [0.7071068, 0, -0.7071068, 0], "+y": [0.7071068, -0.7071068, 0, 0]}[normal]
        parent.append(f'''
      <body name="btn_{bid}" pos="{_f(pos)}" quat="{_f(quat)}">
        <joint name="btn_{bid}" type="slide" axis="0 0 -1" range="0 0.004" stiffness="1200" springref="0" damping="2" armature="0.001"/>
        <geom name="btn_{bid}" type="cylinder" size="0.013 0.004" pos="0 0 0.004" rgba="0.75 0.75 0.72 1" mass="0.01" friction="0.6 0.005 0.0001"/>
        <site name="btn_{bid}" pos="0 0 0.008" size="0.003"/>
      </body>''')
        objects[f"btn_{bid}"] = {"kind": "button", "joint": f"btn_{bid}", "site": f"btn_{bid}", "radius_m": 0.013, "actuation_travel_m": 0.002,
                                 "meaning": meaning}  # fmt: skip
        sensors[f"btn_{bid}"] = {"joint": f"btn_{bid}"}
        displays[f"lamp_{bid}"] = {"fields": {"lamp": {"default": "off", "states": {"off": [], "on": [[f"btn_{bid}", {"rgba": [1.0, 0.85, 0.3, 1]}]]}}}}
        objects[f"lamp_{bid}"] = {"kind": "display", "display": f"lamp_{bid}", "role": "button_lamp", "button": f"btn_{bid}"}

    # The hall: a slab per floor, a front wall with the doorway, landing doors, a call panel.
    for k in range(floors):
        z = k * floor_h
        geoms.append(f'<geom name="slab_{k}" type="box" pos="-2.5 0 {z - 0.1:.4f}" size="2.5 2.5 0.1" rgba="0.62 0.6 0.56 1"/>')
        for side, sgn in (("l", -1), ("r", 1)):  # the hall's front wall, from the doorway's edge out 2.5 m
            geoms.append(f'<geom name="wall_{k}_{side}" type="box" pos="-0.06 {sgn * (half_door + 1.25):.4f} {z + 1.5:.4f}" size="0.06 1.25 1.5" rgba="0.82 0.8 0.75 1"/>')
        geoms.append(f'<geom name="header_{k}" type="box" pos="-0.06 0 {z + 2.55:.4f}" size="0.06 {half_door:.4f} 0.45" rgba="0.82 0.8 0.75 1"/>')
        geoms.append(f'<site name="floor_{k}" pos="-1.0 0 {z:.4f}"/>')
        geoms.append(f'<site name="L{k}_A_sill" pos="0 0 {z:.4f}"/>')
        for side, sgn in (("l", -1), ("r", 1)):
            landing_bodies.append(f'''
    <body name="L{k}_A_door_{side}" pos="-0.14 {sgn * half_door / 2:.4f} {z + 1.055:.4f}">
      <joint name="L{k}_A_door_{side}" type="slide" axis="0 1 0" range="{-half_door - 0.02 if sgn < 0 else -0.01} {0.01 if sgn < 0 else half_door + 0.02}" damping="40"/>
      <geom name="L{k}_A_door_{side}" type="box" size="{PANEL_T / 2} {half_door / 2:.4f} 1.05" rgba="0.55 0.57 0.6 1" mass="25"/>  <!-- 5 mm above the sill -->
    </body>''')
            actuators.append(f'<position name="L{k}_A_door_{side}" joint="L{k}_A_door_{side}" kp="1500" kv="250" forcerange="-100 100"/>')
            sensors[f"L{k}_A_door_force_{side}"] = {"actuator_force": f"L{k}_A_door_{side}"}  # felt through the car door's clutch
        panel = []
        if k < floors - 1:
            button(f"L{k}_up", [-0.12, half_door + 0.65, z + 1.10], "-x", {"type": "hall_call", "floor": f"floor_{k}", "dir": "up"}, panel)
        if k > 0:
            button(f"L{k}_down", [-0.12, half_door + 0.65, z + 1.03], "-x", {"type": "hall_call", "floor": f"floor_{k}", "dir": "down"}, panel)
        landing_bodies.extend(panel)
        objects[f"floor_{k}"] = {"kind": "floor", "label": labels[k], "site": f"floor_{k}"}

    # The shaft walls and pit.
    top = (floors - 1) * floor_h + 3.2
    geoms.append(f'<geom name="pit" type="box" pos="{gap + CAR_D / 2:.4f} 0 -1.6" size="{CAR_D / 2 + 0.1:.4f} {CAR_W / 2 + 0.2:.4f} 0.1" rgba="0.3 0.3 0.3 1"/>')
    geoms.append(f'<geom name="shaft_back" type="box" pos="{gap + CAR_D + 0.15:.4f} 0 {top / 2 - 1.5:.4f}" size="0.05 {CAR_W / 2 + 0.2:.4f} {top / 2 + 1.5:.4f}" rgba="0.4 0.4 0.4 1"/>')

    # The car: floor, walls, ceiling, its doors, its panel, lamps; on a z slide, its counterweight as gravity compensation.
    car_parts, car_panel = [], []
    for side, sgn in (("l", -1), ("r", 1)):
        car_parts.append(f'''
      <body name="car_A_door_{side}" pos="{gap + 0.06:.4f} {sgn * half_door / 2:.4f} 1.055">
        <joint name="car_A_door_{side}" type="slide" axis="0 1 0" range="{-half_door - 0.02 if sgn < 0 else -0.01} {0.01 if sgn < 0 else half_door + 0.02}" damping="40"/>
        <geom name="car_A_door_{side}" type="box" size="{PANEL_T / 2} {half_door / 2:.4f} 1.05" rgba="0.6 0.62 0.65 1" mass="25"/>
      </body>''')
        actuators.append(f'<position name="car_A_door_{side}" joint="car_A_door_{side}" kp="1500" kv="250" forcerange="-100 100"/>')
    cols = max(2, math.ceil(floors / 16))  # at most 16 rows, so the panel stays between about 0.9 and 1.65 m
    for k in range(floors):
        col, row = k % cols, k // cols
        button(f"car_{k}", [gap + 0.6 + 0.045 * col, -CAR_W / 2 + 0.06, 0.95 + 0.045 * row], "+y",
               {"type": "car_floor", "elevator": "car_A", "floor": f"floor_{k}"}, car_panel)
    button("car_open", [gap + 0.6, -CAR_W / 2 + 0.06, 0.85], "+y", {"type": "door_open", "elevator": "car_A"}, car_panel)
    button("car_close", [gap + 0.645, -CAR_W / 2 + 0.06, 0.85], "+y", {"type": "door_close", "elevator": "car_A"}, car_panel)
    lamps = "".join(f'<geom name="pos_A_{k}" type="box" pos="{gap + 0.09:.4f} {-0.3 + 0.6 * k / max(floors - 1, 1):.4f} 2.25" size="0.005 0.02 0.02" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0" mass="0"/>'
                    for k in range(floors))  # fmt: skip
    displays["pos_A"] = {"fields": {"floor": {"default": labels[0], "states": {labels[k]: [[f"pos_A_{k}", {"rgba": [1, 0.4, 0.1, 1]}]] for k in range(floors)}}}}
    objects["pos_A"] = {"kind": "display", "display": "pos_A", "role": "car_position", "elevator": "car_A"}
    jamb = half_door + 0.04
    car = f'''
    <body name="car_A" pos="0 0 0" gravcomp="1">
      <joint name="car_A_z" type="slide" axis="0 0 1" range="-0.5 {travel + 0.5:.3f}" damping="2000"/>
      <geom name="car_A_floor" type="box" pos="{gap + CAR_D / 2:.4f} 0 -0.05" size="{CAR_D / 2:.4f} {CAR_W / 2:.4f} 0.05" rgba="0.45 0.42 0.38 1" mass="600"/>
      <geom name="car_A_back" type="box" pos="{gap + CAR_D - 0.02:.4f} 0 {CAR_H / 2:.4f}" size="0.02 {CAR_W / 2:.4f} {CAR_H / 2:.4f}" rgba="0.7 0.68 0.62 1" mass="50"/>
      <geom name="car_A_side_l" type="box" pos="{gap + CAR_D / 2:.4f} {-CAR_W / 2 + 0.02:.4f} {CAR_H / 2:.4f}" size="{CAR_D / 2:.4f} 0.02 {CAR_H / 2:.4f}" rgba="0.7 0.68 0.62 1" mass="50"/>
      <geom name="car_A_side_r" type="box" pos="{gap + CAR_D / 2:.4f} {CAR_W / 2 - 0.02:.4f} {CAR_H / 2:.4f}" size="{CAR_D / 2:.4f} 0.02 {CAR_H / 2:.4f}" rgba="0.7 0.68 0.62 1" mass="50"/>
      <geom name="car_A_front_l" type="box" pos="{gap + 0.1:.4f} {-(CAR_W / 2 + half_door) / 2:.4f} {CAR_H / 2:.4f}" size="0.02 {(CAR_W / 2 - half_door) / 2:.4f} {CAR_H / 2:.4f}" rgba="0.7 0.68 0.62 1" mass="20"/>
      <geom name="car_A_front_r" type="box" pos="{gap + 0.1:.4f} {(CAR_W / 2 + half_door) / 2:.4f} {CAR_H / 2:.4f}" size="0.02 {(CAR_W / 2 - half_door) / 2:.4f} {CAR_H / 2:.4f}" rgba="0.7 0.68 0.62 1" mass="20"/>
      <geom name="car_A_ceiling" type="box" pos="{gap + CAR_D / 2:.4f} 0 {CAR_H + 0.02:.4f}" size="{CAR_D / 2:.4f} {CAR_W / 2:.4f} 0.02" rgba="0.9 0.9 0.88 1" mass="40"/>
      {lamps}
      <site name="car_A_floor" pos="{gap + CAR_D / 2:.4f} 0 0"/>
      <site name="car_A_sill" pos="{gap:.4f} 0 0" quat="0 0 0 1"/>
      <site name="car_A_beam_1a" pos="{gap + 0.13:.4f} {-jamb:.4f} 0.3"/><site name="car_A_beam_1b" pos="{gap + 0.13:.4f} {jamb:.4f} 0.3"/>
      <site name="car_A_beam_2a" pos="{gap + 0.13:.4f} {-jamb:.4f} 1.0"/><site name="car_A_beam_2b" pos="{gap + 0.13:.4f} {jamb:.4f} 1.0"/>
      {"".join(car_parts)}{"".join(car_panel)}
    </body>'''
    actuators.append('<velocity name="car_A_motor" joint="car_A_z" kv="40000" forcerange="-30000 30000"/>')  # the drive: a speed command

    xml = f'''<mujoco model="elevator_example_{seed}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" impratio="10" cone="elliptic" integrator="implicitfast"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.5 0.5 0.5"/></visual>
  <default><geom friction="0.8 0.005 0.0001"/></default>
  <worldbody>
    <light pos="-2 0 {top:.2f}" dir="0.3 0 -1" diffuse="0.6 0.6 0.6"/>
    {"".join(geoms)}
    {"".join(landing_bodies)}
    {car}
  </worldbody>
  <actuator>{"".join(actuators)}</actuator>
</mujoco>
'''
    out = Path(out_dir)
    (out / "world.xml").write_text(xml)
    shutil.copy(HERE / "dynamics.py", out / "dynamics.py")
    landings = {f"floor_{k}": {"sill_site": f"L{k}_A_sill", "doors": {"joints": [f"L{k}_A_door_l", f"L{k}_A_door_r"], "closed": [0.0, 0.0],
                                                                      "open": [-half_door, half_door]}} for k in range(floors)}  # fmt: skip
    objects["car_A"] = {"kind": "elevator", "joint": "car_A_z", "floor_site": "car_A_floor", "sill_site": "car_A_sill",
                        "doors": {"joints": ["car_A_door_l", "car_A_door_r"], "closed": [0.0, 0.0], "open": [-half_door, half_door]},
                        "beams": ["car_A_beam_1", "car_A_beam_2"], "landings": landings}  # fmt: skip
    sensors.update({"car_A": {"joint": "car_A_z"}, "car_A_load": {"load": "car_A"}, "car_A_door": {"joint": "car_A_door_r"},
                    "car_A_door_force_l": {"actuator_force": "car_A_door_l"}, "car_A_door_force_r": {"actuator_force": "car_A_door_r"},
                    "car_A_beam_1": {"beam": ["car_A_beam_1a", "car_A_beam_1b"]}, "car_A_beam_2": {"beam": ["car_A_beam_2a", "car_A_beam_2b"]}})  # fmt: skip
    ann = {
        "control_hz": 50,
        "dynamics": "dynamics.py",
        "params": {**run, "floor_z": [round(k * floor_h, 4) for k in range(floors)], "floor_labels": labels},
        "actuators": [a.split('name="')[1].split('"')[0] for a in actuators],
        "sensors": sensors,
        "displays": displays,
        "objects": objects,
        "faults": [{"name": "missed_press", "rate_param": "fault_missed_press_rate", "recovery": {"presses": 3},
                    "description": "a press is not registered (a worn contact); the next one usually is"}],  # fmt: skip
    }
    import yaml

    (out / "annotations.yaml").write_text(yaml.safe_dump(ann, sort_keys=False))
    return {"build": build, "run": {k: v for k, v in run.items()}}
