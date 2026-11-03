# onboard/drivers/base: the hb1 base driver ChefBot runs

`driver.py` is ChefBot's own Redis loop around the TidyBot++ `Vehicle`. It keeps the Redis contract of ZitiBot's
`controllers/tidybot_base/redis_driver.py` and adds a command timeout, a heartbeat, and a kill that exits
(see its docstring). Tests: `chefbot/tests/onboard/test_base_driver.py`.

Copied unchanged from `~/OpenSai/ZitiBot/controllers/tidybot_base/` at ZitiBot commit
`7530046c890e6bcd04ef2007cc74f825e44e97ae` (that folder had no local changes on 2026-11-03):

| File | sha256 |
|---|---|
| `base_controller.py` (Vehicle: casters over CAN with phoenix6, Ruckig trajectory, odometry) | `ebde293233d177f32dddd4541e3da927b5bbf01b849e04c5388dbdd8eb9d1923` |
| `constants.py` (base geometry, encoder offsets for Base #3) | `0c89f0461e0a86aafca00652590a45777baa5ad901f47cd81f6c6e170926233d` |
| `utils.py` (PID file) | `749f8bc018758cf6db7e8bec3b95a7ce01c31bb49521a539ff5904d984e2aa5a` |

Do not edit these three; change `driver.py`. Runs in the robot's `zitibot` env (phoenix6, ruckig, threadpoolctl, redis).
