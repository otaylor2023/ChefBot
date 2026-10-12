# Isaac backend (placeholder)

Not implemented. An Isaac backend has the same shape as `envs/robocasa/`:

- `env.py`: `IsaacEnv(Env)`, low-level only (`get_state`, `set_base_velocity`, `wait_for_next_tick`, ...).
- `detector.py`: a `Detector` that takes ChefBot labels and maps them to Isaac prim names internally.
- `build_map.py`: scene -> `semantic_map.yaml` + `grid.npy` / `grid.yaml`.
- `serve.py`: `RobotServer(Robot(IsaacEnv()), detector)`.

Nothing under `chefbot/agent/` or `chefbot/robot/` changes.
