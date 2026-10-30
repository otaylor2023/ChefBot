# The checks

`./lab build` runs these on every building (the code: `chefbot/generators/elevator/checks.py` and `probes.py`; you
may read them). Each scenario is a fresh run of your scene with your controller, sandboxed, and the lab's probes:

- **finger**: a fingertip (7 mm radius) on a mount, pushed into a button along the button's axis by a motor;
- **cart**: a 40 kg box (0.5 x 0.4 m) on four 5 cm ball wheels, driven at up to 0.4 m/s: a small robot's footprint;
- **block**: a box (0.3 x 0.12 x 0.8 m) put in a doorway.

The trip goes from the lowest floor to the highest (the longest ride). A building **passes** when every scenario
does.

| check | what happens | passes when |
|---|---|---|
| structure | the scene loads, the controller starts | the annotations name what the contract requires and the model has it; every car has a `car_position` display; the required parameters are there; at most 60 floors |
| trip | faults off. A 0.4 N touch on the hall call (or destination key); a 6 N press; wait; the cart drives in; the car's floor button; the ride; the cart drives out | the touch does nothing; the press registers (and its lamp lights, unless a car opens there at once); a car levels and opens within `max_wait_s` + 2 s; sill gap within the declared gap + 5 mm and 32 mm; step within `leveling_tol_m` + 1 mm and 13 mm; the cart gets in and out; the floor button registers; peak speed within 1.1 x `rated_speed_mps`, acceleration within 1.2 x `accel_mps2` + 0.1, jerk within 1.5 x `jerk_mps3` + 0.3 (smoothed over 60 / 120 ms); the cart rides along; the car levels at the destination and its position display shows that floor; doors never push the cart harder than 135 N |
| doors | faults off: a car called to the top floor, left alone; then again with `dwell_s` overridden (+3 s) | door opening and closing times (5% to 95%) within 25% (or 0.5 s) of `door_open_s`/`door_close_s`; dwell within 20% (or 0.6 s) of `dwell_s`, also with the override |
| obstruction | a car open at the lowest floor; as its doors start closing, the block goes in the doorway | the doors reopen within 6 s, never close on the block, and push it no harder than 1.15 x `door_force_limit_N` and about 150 N (averaged over 0.2 s; impacts are reported, not checked) |
| hold_open | the door-open button held for `dwell_s` + 4 s | the doors stay open (skipped when a car has no door-open button) |
| calls | hall calls at up to three floors at once (the first pressed twice) | every one is served (a car leveled and open there) |
| faults | each declared fault alone, its rate set to 1; a call, recovered as the fault declares | the fault fires (an event), and a car still comes |
| speed | all the above | simulated faster than real time |

**In every scenario:** a car's doors are open (over 10%) only within 75 mm of a floor and below 0.3 m/s; a landing's
doors are open only with a car within 75 mm there; with every fault rate at 0, no fault events.

**Ranges:** every parameter in `params.json` inside `ranges.yaml`, every range with a `why`, and ranges for the
required parameters.

Notes from building the lab:
- MuJoCo never filters contacts with the world body: a landing door resting on its floor slab scrapes it. Real doors
  run a few mm above the sill.
- A position actuator's damping acts against the absolute velocity, so a car on one lags its target badly at speed;
  real drives are speed-controlled. The example drives the car with a velocity actuator.
- `gravcomp` balances a body only, not its children (doors, buttons): a counterweight that misses them leaves the car
  low. Hold the floor with feedback (the example uses integral action).
- Doors "commanded closed" are not doors closed: confirm with the door position (a door-closed contact), or a block at
  the meeting edge is never noticed.
