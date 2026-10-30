# The elevator generator contract

A **generator** is a directory (`generators/<name>/`) with `generate.py`, `ranges.yaml`, and whatever else it
needs. Each seed makes **one building with elevators**, as a plain-MuJoCo scene: its floors, hallways, shafts, cars,
doors, buttons, lights, displays, and its **controller** (the scene's dynamics). Different seeds should give the
variety of real buildings: building types, sizes, numbers of floors (at most 60) and cars, door and panel designs,
destination dispatch, controllers, timings, faults. The lab runs the generator in a sandbox (no network, only its own
directory and the output directory, at most {GEN_TIMEOUT_S} s and {MAX_ASSET_MB} MB per building).

```python
def generate(seed: int, out_dir: str, overrides: dict | None = None) -> dict:
    """Write the scene into out_dir. `overrides` fixes build parameters (e.g. {"floors": 20}); the rest are drawn
    from your distributions. Return {"build": {...}, "run": {...}}: every parameter, in physical units."""
```

Deterministic per (seed, overrides). Its Python: {GEN_PACKAGES}. The lab writes what it returns to params.json.

## The scene (out_dir)

```
world.xml          MuJoCo {MUJOCO_VERSION}, <compiler angle="radian"/>, z up; mesh and texture paths relative to out_dir
annotations.yaml   what the parts are, what the controller senses, sets and shows, the run parameters, the faults
dynamics.py        the controller
```

The lab adds its test probes to the scene (names starting `lab_`). Keep at least 2.5 m of clear floor in front of
every landing's doors (a cart starts there). Timestep at most 2 ms; a 60-floor building must still simulate faster
than real time.

### world.xml

- **Cars**: a body per car on **one slide joint along z**, driven by actuators you define. Car doors are bodies inside
  the car; landing doors are bodies in the building; each door panel on its own joint, driven by your actuators.
  Motors need realistic `forcerange`: the lab measures the force doors put on an obstacle.
- **Buttons**: a body per button on **one slide joint along the button's axis**, a spring (`stiffness`,
  `springref`) and a `range` for its travel. The lab's test finger presses it like a fingertip.
- Geometry, materials, signage, Braille, lighting, finishes: yours. Collision geometry must follow what a robot will
  touch: floors, sills and the gap between them, door edges, button faces, walls, handrails.

### annotations.yaml

The format of a plain-MuJoCo scene (`chefbot/environments/mujoco/scene.py`), with these objects:

```yaml
control_hz: 50
dynamics: dynamics.py
params: {...}                    # the run parameters (below): what the controller gets by default
actuators: [car_A_motor, car_A_door, L0_A_door, ...]
sensors:                         # what the controller may read, by name (scene.py: joint, actuator_force, beam, load)
  car_A: {joint: car_A_z}
  car_A_load: {load: car_A}
  car_A_door: {joint: car_A_door_l}
  car_A_door_force: {actuator_force: car_A_door}
  car_A_beam_1: {beam: [car_A_beam_1a, car_A_beam_1b]}
  btn_L0_up: {joint: btn_L0_up}
displays: {...}                  # scene.py's display format
objects:
  floor_0:   {kind: floor, label: "L", site: floor_0}      # site: on the landing floor in front of the elevators, z up
  car_A:     {kind: elevator, joint: car_A_z, floor_site: car_A_floor, sill_site: car_A_sill,
              doors: {joints: [car_A_door_l, car_A_door_r], closed: [0, 0], open: [-0.45, 0.45]},
              beams: [car_A_beam_1, car_A_beam_2],          # sensor names
              landings: {floor_0: {sill_site: L0_A_sill, doors: {joints: [...], closed: [...], open: [...]}}, ...}}
  btn_L0_up: {kind: button, joint: btn_L0_up, site: btn_L0_up, radius_m: 0.012, actuation_travel_m: 0.002,
              meaning: {type: hall_call, floor: floor_0, dir: up}}
  pos_A:     {kind: display, display: pos_A, role: car_position, elevator: car_A}
  lamp_L0_up: {kind: display, display: lamp_L0_up, role: button_lamp, button: btn_L0_up}
faults:
  - {name: missed_press, rate_param: fault_missed_press_rate, recovery: {presses: 3}, description: "..."}
```

- **floor**: every floor; `site` on the landing floor in front of the elevators.
- **elevator** (one per car): `floor_site` at the centre of the car floor; `sill_site` at the centre of the car sill
  with **x pointing out of the car**; each landing's `sill_site` at the landing sill's centre with **x pointing into
  the car**. Door `closed`/`open` are the door joints' positions. `beams`: sensor names of the light curtain or photo
  eyes across the car doorway. `landings`: every floor the car serves.
- **button** (every button): `site` at the face centre with **z pointing out of the face**; `actuation_travel_m`: how
  far in it makes contact (the lab logs a *physical press* there). `meaning.type`: `hall_call` (`floor`, `dir`),
  `car_floor` (`elevator`, `floor`), `door_open`, `door_close` (`elevator`), `alarm`, `dispatch_key` (destination
  dispatch: a key or touchscreen button on the keypad `at` a floor that asks for the destination `to`; a keypad has
  one per destination it offers), or `other`. With destination dispatch the car is assigned by the controller (show
  which on a display) and the car may have no floor buttons.
- **display**: a `car_position` display must have a field `floor` whose states are the floors' labels (on arrival it
  must show the car's floor); a `button_lamp`, a field `lamp` with states `on` and `off`. Lanterns, screens, kiosks:
  any role you like.
- **faults**: every way the elevator misbehaves on purpose. Each has a run parameter giving its rate (0 to 1, a
  probability per opportunity you define) and the `recovery` a person would use: `{presses: n}` (press up to n
  times), `{hold_s: s}` (press and hold), or `{wait_s: s}` (it works within s).

### dynamics.py: the controller

```python
class Dynamics:
    def __init__(self, annotations: dict, params: dict, seed: int): ...
    def step(self, t: float, sensed: dict) -> dict: ...
```

The environment calls `step` at `control_hz` in a sandboxed process: no network, no files, no simulator. `params`
are the run parameters (annotations' `params` with any overrides); `seed` drives all of its randomness (same seed,
same behaviour). It sees only `sensed` (each sensor: a joint `{"q", "v"}`, a force, a beam's bool, a load in kg), and
returns:

```python
{"ctrl": {actuator: value},
 "displays": {display: {field: state}},
 "events": [{"type": "registered", "button": id},           # a press the elevator accepted (required)
            {"type": "fault", "name": n, "detail": "..."},   # a fault firing (required)
            {"type": "assigned", "button": id, "elevator": "car_A"}, ...]}   # anything else useful
```

Everything about how the elevators behave is the controller's: which presses count, dispatch, motion profiles,
leveling, door timing, nudging, reopening, lamps, screens, faults, traffic. A physical press (button travel) and the
elevator accepting it (`registered`) are different things.

## Parameters

`generate()` returns `{"build": {...}, "run": {...}}`. **Build** parameters need a new building (floors, cars, door
type, panel style, sizes, materials, building type). **Run** parameters are the annotations' `params`: the controller
reads them, and they can change per episode without regenerating. Required run parameters:

| key | meaning |
|---|---|
| `max_wait_s` | the longest a call waits for a car, with no faults and no other traffic |
| `door_open_s`, `door_close_s` | time for the doors to open fully / close fully |
| `dwell_s` | how long doors stay fully open after arriving (no one in the doorway) |
| `rated_speed_mps`, `accel_mps2`, `jerk_mps3` | the car's motion limits |
| `leveling_tol_m` | how far from the floor the car may stop |
| `door_force_limit_N` | the most the closing doors push on an obstacle |
| `fault_<name>_rate` | one per declared fault: its rate |

Required build parameters: `floors`, `cars`, `sill_gap_m` (the horizontal gap between sills), `building_type`.

`ranges.yaml` gives every parameter (build and run) its range or values, distribution, and **why** (codes,
catalogs, measurements, with sources). Each building's parameters must fall inside.
