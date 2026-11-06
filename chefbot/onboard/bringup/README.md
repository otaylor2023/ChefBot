# Robot computer bring-up (tidybot01)

The robot's drivers are the existing ones in `~/OpenSai`, used in place. This folder only starts
them and the onboard server.

## Once

```sh
git clone <ChefBot> ~/ChefBot && cd ~/ChefBot
```

The server runs in `~/chefbot_calib/env` (launch.sh's `CHEFBOT_PYTHON`): numpy, redis, pyrealsense2, websockets,
msgpack, msgpack-numpy, opencv; no planning, IK or simulator.

## Each session

1. Franka Desk: unlock the joints, FCI on.
2. The NUC on SRC wifi (`sudo nmcli con up SRC`): the kitchen's mocap streams only there.
3. `cd ~/ChefBot && chefbot/onboard/bringup/launch.sh`: the Franka arm driver (asks for sudo once), OpenSai holding
   the arm, the gripper and base drivers, the mocap publisher (`mocap.sh`: `chefbot-tidy::pos` / `::ori` on Redis),
   then the onboard server. Ctrl-C stops them all, the arm driver first (the brakes hold the arm); it says whether
   the arm driver stopped. While it runs, **f** floats the arm (OpenSai's `float_controller`: no stiffness, light
   damping; move it by hand), **h** holds it where it is, **r** takes it to the ready position (interface/workspace.py
   READY; from floating too; refused if the straight way passes the pole's keep-out); **g** gives the base to the gamepad (the base
   driver stops, the teleop below starts; Start on the gamepad begins) and **b** gives it back to its driver (its
   odometry starts over). `--no-arm-driver` if one is already running;
   `arm.sh` is just the arm driver and OpenSai; `float.sh` just the arm driver (the arm floating on zero torques);
   `base.sh` just the base (`--no-arm`: the base driver, mocap and the server; refused while an arm driver runs unless
   `--allow-arm`); `mocap.sh` alone runs just the mocap publisher, printing each frame.
4. To drive the base by hand: `gamepad.sh` (the TidyBot++ gamepad teleop, `onboard/drivers/base/gamepad_teleop.py`,
   through `teleop.py`, in the zitibot env, at the `teleop` speed profile; `--profile NAME` for another, never over the
   base cap). `--one-axis`: one motion at a time (the mocap marker calibration's drives, `calibration/base.py`);
   `--echo`: the stick and the command it would send, no base. Not while a base driver runs (except `--echo`): use
   launch.sh's **g**, start launch.sh with `--no-base`, or use it alone.

## Navigation on the robot itself

`scripts/real_nav.py` and `scripts/real_park.py` can run on the NUC against the server there (`--url
ws://127.0.0.1:8765`), so the drive loop never crosses the wifi (PR #2's kitchen runs: 60 of 60 trips). They are the
robot layer (robot/, embodiments/), not onboard code, so they need its dependencies (mujoco, mink, ...), which the
server's env (`~/chefbot_calib/env`) doesn't have: run them in an env that does (on 2026-11-05, Jack's
`~/miniforge3/envs/chefbot-movement`), with the repo on PYTHONPATH. `docs/kitchen-runbook.md` has the commands.

Then, read-only, on the robot: `python -m chefbot.onboard.check` (which drivers are up, nothing moves).
From the laptop: `chefbot/environments/real/up.sh` (the tunnel and the vision server), then
`python -m chefbot.environments.real.check`.

## Safety

- The server holds the arm where it is and stops the base when its client goes quiet for 0.3 s
  (`server.watchdog_s`) or disconnects. Keep a hand on the e-stop anyway.
- Base speed never exceeds the robot's hard base cap, `embodiments/tidybot_fr3/robot.yaml` `control.limits` (its one
  copy, read by `interface/limits.py`): the driver runs at it, the server and the gamepad are held to it, and a speed
  profile over it is refused wherever the base can move.
- `launch_gripper.sh` has the robot computer's sudo password in it; it stays out of this repo.
