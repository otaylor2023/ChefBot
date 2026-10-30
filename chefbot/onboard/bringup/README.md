# Robot computer bring-up (tidybot01)

The robot's drivers are the existing ones in `~/OpenSai`, used in place. This folder only starts
them and the onboard server.

## Once

```sh
git clone <ChefBot> ~/ChefBot && cd ~/ChefBot
~/miniforge3/bin/conda create -n chefbot python=3.11
~/miniforge3/envs/chefbot/bin/pip install -e ".[onboard]"
```

`[onboard]` is the server's few dependencies (redis, pyrealsense2; no planning, IK or simulator).

## Each session

1. Franka Desk: unlock the joints, FCI on.
2. The Franka arm driver (real-time torque loop), as for ZitiBot, from `~/OpenSai/drivers/FrankaPanda/redis_driver`.
3. The mocap publisher (Motive / NatNet), so `tidybot01::pos` and `tidybot01::ori` appear on Redis.
4. `cd ~/ChefBot && chefbot/onboard/bringup/launch.sh`: OpenSai, the gripper and base drivers, then the
   onboard server. Ctrl-C stops them all.

Then, read-only, on the robot: `python -m chefbot.onboard.check` (which drivers are up, nothing moves).
From the laptop: `ssh -L 8765:localhost:8765 tidybot01`, then `python -m chefbot.environments.real.check`.

## Safety

- The server holds the arm where it is and stops the base when its client goes quiet for 0.3 s
  (`server.watchdog_s`) or disconnects. Keep a hand on the e-stop anyway.
- Base speed never exceeds the hb1 driver's own `--max-vel-xy` / `--max-vel-yaw` (launch.sh's
  `MAX_VEL_XY`, `MAX_VEL_YAW`; keep them equal to `onboard/config.yaml`).
- `launch_gripper.sh` has the robot computer's sudo password in it; it stays out of this repo.
