# Real robot backend (placeholder)

Not implemented. Planned: mocap for the base pose, TidyBot++ and FR3 drivers,
and an open-vocabulary detector (detection + depth -> world point).

`RealEnv.wait_for_next_tick(dt)` will likely be a no-op, which is only safe if
something else paces the control loop, e.g. `get_state()` blocking until a new
mocap sample arrives. If nothing blocks, the control loop in
`chefbot/robot/robot.py` spins at CPU speed.
