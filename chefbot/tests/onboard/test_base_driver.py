"""onboard/drivers/base/driver.py's loop against a fake Vehicle and a fake Redis (no CAN, no phoenix6)."""

import numpy as np
import pytest

from chefbot.onboard.drivers.base import driver as D


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = str(value)

    def incr(self, key):
        self.set(key, int(self.values.get(key, 0)) + 1)


class FakeOtg:
    max_velocity = [0.125, 0.125, 0.4]


class FakeVehicle:
    def __init__(self):
        self.x, self.dx = np.zeros(3), np.zeros(3)
        self.max_vel = np.array([0.125, 0.125, 0.4])
        self.otg_inp = FakeOtg()
        self.commands = []

    def set_target_position(self, p):
        self.commands.append(("position", np.asarray(p, dtype=float).tolist()))

    def set_target_velocity(self, v):
        self.commands.append(("velocity", np.asarray(v, dtype=float).tolist()))


@pytest.fixture
def rig():
    now = [100.0]
    # Left by an earlier run: an old command sequence and a goal far away.
    redis = FakeRedis({D.CMD_SEQ: "41", D.DESIRED: "[5.0, 5.0, 0.0]", D.STOP: "stop"})
    vehicle = FakeVehicle()
    loop = D.DriverLoop(vehicle, redis, command_timeout_s=0.5, clock=lambda: now[0])
    return loop, vehicle, redis, now


def command(redis, pose):
    redis.set(D.DESIRED, D._list(pose))
    redis.incr(D.CMD_SEQ)


def test_startup_ignores_whatever_was_left_in_redis(rig):
    loop, vehicle, redis, now = rig
    assert redis.values[D.DESIRED] == "[0.0, 0.0, 0.0]" and redis.values[D.STOP] == "ok"
    for _ in range(5):
        now[0] += 0.01
        loop.step()
    assert all(kind == "velocity" for kind, _ in vehicle.commands) and loop.mode == "idle"


def test_a_fresh_command_drives_and_a_silent_client_brakes_within_the_timeout(rig):
    loop, vehicle, redis, now = rig
    command(redis, [0.3, 0.0, 0.0])
    loop.step()
    assert vehicle.commands[-1] == ("position", [0.3, 0.0, 0.0]) and loop.mode == "moving"
    now[0] += 0.4  # 0.4 s with no new command: still within the timeout
    loop.step()
    assert vehicle.commands[-1][0] == "position"
    now[0] += 0.2  # 0.6 s: expired
    vehicle.x = np.array([0.1, 0.0, 0.0])
    loop.step()
    assert vehicle.commands[-1] == ("velocity", [0.0, 0.0, 0.0]) and loop.mode == "idle"
    assert redis.values[D.DESIRED] == "[0.1, 0.0, 0.0]"  # held where it is, the old goal dropped
    now[0] += 5.0
    loop.step()
    assert vehicle.commands[-1][0] == "velocity"  # and it does not come back on its own


def test_stop_brakes_resets_once_still_and_ends_the_command(rig):
    loop, vehicle, redis, now = rig
    command(redis, [0.3, 0.0, 0.0])
    loop.step()
    redis.set(D.STOP, "stop")
    vehicle.dx = np.array([0.05, 0.0, 0.0])
    now[0] += 0.01
    loop.step()
    assert vehicle.commands[-1][0] == "velocity" and redis.values[D.STOP] == "stop"
    vehicle.dx = np.zeros(3)
    now[0] += 0.01
    loop.step()
    assert redis.values[D.STOP] == "ok"
    now[0] += 0.01
    loop.step()
    assert vehicle.commands[-1][0] == "velocity"  # the pre-stop command is not resumed


def test_heartbeat_and_state_publish_every_tick_even_while_braking(rig):
    loop, vehicle, redis, now = rig
    beats = []
    for _ in range(3):
        now[0] += 0.01
        loop.step()
        beats.append(int(redis.values[D.HEARTBEAT]))
    assert beats == sorted(set(beats)) and D.POSE in redis.values and D.VEL in redis.values


def test_kill_exits_the_loop(rig):
    loop, _, redis, _ = rig
    redis.set(D.KILL, "kill")
    assert loop.step() is False and loop.mode == "killed"


def test_a_garbled_goal_brakes(rig):
    loop, vehicle, redis, _ = rig
    redis.set(D.DESIRED, "[nan, 0, 0]")
    redis.incr(D.CMD_SEQ)
    loop.step()
    assert vehicle.commands[-1][0] == "velocity"


def test_speed_scale_never_exceeds_the_driver_limits(rig):
    loop, vehicle, redis, _ = rig
    redis.set(D.MAX_VEL_SCALE, "3.0")
    loop.step()
    assert loop.scale == 1.0
    redis.set(D.MAX_VEL_SCALE, "0.5")
    loop.step()
    assert vehicle.otg_inp.max_velocity == pytest.approx([0.0625, 0.0625, 0.2])


@pytest.mark.parametrize("goal", [[0.3, 0.0, 0.0], [0.3, 0.3, 0.0], [-0.1, 0.25, 1.0], [0.0, -0.4, 0.0]])
def test_the_planar_speed_limit_holds_along_any_direction(rig, goal):
    # Per-axis limits of 0.125 let a 45 deg diagonal reach 0.177 m/s (0.167 measured in HT-3b).
    loop, vehicle, redis, _ = rig
    command(redis, goal)
    loop.step()
    vx, vy, w = vehicle.otg_inp.max_velocity
    d = np.array(goal[:2]) / np.linalg.norm(goal[:2])
    # The fastest the synchronized trajectory can go along d: each axis at most its limit.
    along = min(vx / abs(d[0]) if abs(d[0]) > 1e-9 else np.inf, vy / abs(d[1]) if abs(d[1]) > 1e-9 else np.inf)
    assert along <= 0.125 + 1e-9 and along >= 0.125 * 0.99
    assert w == pytest.approx(0.4)
