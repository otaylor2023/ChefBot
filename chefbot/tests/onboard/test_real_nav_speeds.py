"""real_nav's base speeds: from the speed profiles under the robot's one hard cap (interface/limits.py), refused (not
clipped) when over the profile, the cap, or what the robot computer reports; the measured stop has one copy too."""

import pytest

from chefbot import embodiments
from chefbot.interface.limits import base_cap, base_stop
from chefbot.robot.nav.speed import StopModel
from chefbot.scripts.real_nav import base_speeds, check_server_cap

ROBOT = embodiments.load("tidybot_fr3")


def test_a_profile_gives_its_base_speeds():
    assert base_speeds(ROBOT, "normal", None, None) == (0.2, 0.375, "normal")


def test_the_default_is_the_robots_profile():
    assert base_speeds(ROBOT, None, None, None)[2] == ROBOT.raw["control"]["speed"]


def test_a_profile_over_the_cap_is_refused():
    with pytest.raises(SystemExit, match="refusing"):
        base_speeds(ROBOT, "fast", None, None)


def test_lower_speeds_are_allowed_higher_ones_refused():
    assert base_speeds(ROBOT, "normal", 0.1, 0.2)[:2] == (0.1, 0.2)
    with pytest.raises(SystemExit, match="--speed"):
        base_speeds(ROBOT, "normal", 0.21, None)
    with pytest.raises(SystemExit, match="--yaw-speed"):
        base_speeds(ROBOT, "normal", None, 0.5)


def test_a_robot_reporting_a_lower_cap_is_refused():
    cap = base_cap("tidybot_fr3")["vel"]
    check_server_cap({"base_max_vel_xy": cap[0], "base_max_vel_yaw": cap[2]}, cap[0], cap[2])
    with pytest.raises(ValueError, match="deploy"):
        check_server_cap({"base_max_vel_xy": 0.125, "base_max_vel_yaw": 0.4}, 0.2, 0.375)
    with pytest.raises(ValueError, match="reports no"):
        check_server_cap({}, 0.1, 0.1)


def test_the_measured_stop_is_the_embodiments():
    assert StopModel.from_limits(base_stop("tidybot_fr3")) == StopModel(tau_s=0.333, brake_mps2=0.25)


def test_the_server_reports_the_cap_and_the_measured_stop_in_hello():
    from chefbot.onboard.server import OnboardServer

    class _Cams:
        cameras: dict = {}

        def close(self):
            pass

    config = {"server": {"host": "127.0.0.1", "port": 0, "state_hz": 50, "watchdog_s": 0.3, "camera_idle_s": 10}}
    server = OnboardServer(None, _Cams(), config)
    server.start()
    try:
        cap = base_cap("tidybot_fr3")
        assert server.limits == {"base_max_vel_xy": cap["vel"][0], "base_max_vel_yaw": cap["vel"][2],
                                 "base_max_accel_xy": cap["accel"][0], **base_stop("tidybot_fr3")}  # fmt: skip
    finally:
        server.shutdown()
