# onboard/drivers/base: the hb1 base driver ChefBot runs

The TidyBot++ base driver (casters over CAN with phoenix6, a Ruckig trajectory, wheel odometry) on the robot's
Redis, copied unchanged from ZitiBot (`otaylor2023/ZitiBot`, `controllers/tidybot_base/`) at commit
`7530046c890e6bcd04ef2007cc74f825e44e97ae`, so ChefBot runs its own copy instead of ZitiBot's checkout on the NUC
(`onboard/bringup/launch.sh` starts it). Changes for ChefBot go here, in `driver.py`.

| File | From | sha256 (as copied) |
|---|---|---|
| `driver.py` (the Redis loop: hb1::current_pose / ::current_vel out, hb1::desired_pose / ::stop / ::kill / ::max_vel_scale in) | `redis_driver.py` | `917f067435bd2948bdeabbedc37458047373f2fc2178ee7d11e3c230068c9ff6` |
| `base_controller.py` (Vehicle: casters over CAN with phoenix6, Ruckig trajectory, odometry) | same name | `ebde293233d177f32dddd4541e3da927b5bbf01b849e04c5388dbdd8eb9d1923` |
| `constants.py` (base geometry, encoder offsets for Base #3) | same name | `0c89f0461e0a86aafca00652590a45777baa5ad901f47cd81f6c6e170926233d` |
| `utils.py` (PID file) | same name | `749f8bc018758cf6db7e8bec3b95a7ce01c31bb49521a539ff5904d984e2aa5a` |
| `gamepad_teleop.py` (drive the base by gamepad; run by `onboard/bringup/gamepad.sh` at the driver's limits) | same name | `ef30974ff4c746ff4db5cfda2975912b81270b30a16d52cec070ce78363db7ef` |

Keep `base_controller.py`, `constants.py`, `utils.py` and `gamepad_teleop.py` as copied. Runs in the robot's `zitibot` env (phoenix6, ruckig,
threadpoolctl, redis), from this folder (the files import each other by name):

    cd chefbot/onboard/drivers/base && ~/miniforge3/envs/zitibot/bin/python driver.py --max-vel-xy 0.125 --max-vel-yaw 0.4
