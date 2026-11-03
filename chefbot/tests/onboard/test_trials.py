"""scripts/real_trials: footprint geometry, the violation checker, and a short session against onboard/sim.py."""

import math

import numpy as np
import pytest

from chefbot.envs.real.env import RealEnv
from chefbot.onboard.sim import SimRobot
from chefbot.scripts import real_trials as T
from chefbot.scripts.real_map import box_polygon

FENCE = box_polygon((0.0, 0.0), (2.0, 2.0))
BOX = box_polygon((0.6, 0.0), (0.2, 0.2))  # 0.5 .. 0.7 in x


def test_the_arm_box_turns_with_the_base():
    boxes = T.footprint_boxes((0.4, 0.3), [(0.2, 0.1, 0.4)])  # arm box centred 0.4 m ahead
    facing_x = T.placed(boxes, (0.0, 0.0, 0.0))[1]
    facing_y = T.placed(boxes, (0.0, 0.0, math.pi / 2))[1]
    assert np.mean(facing_x, axis=0) == pytest.approx((0.4, 0.0))
    assert np.mean(facing_y, axis=0) == pytest.approx((0.0, 0.4))
    assert T.enclosing_radius(boxes) == pytest.approx(math.hypot(0.5, 0.05))


def test_the_checker_catches_the_arm_reaching_an_obstacle_the_base_clears():
    boxes = T.footprint_boxes((0.4, 0.3), [(0.2, 0.1, 0.4)])
    base_only = T.footprint_boxes((0.4, 0.3))
    pose_facing_box = (0.1, 0.0, 0.0)  # base front at 0.3, arm box front at 0.6: inside the box (0.5 .. 0.7)
    assert T.check_poses([pose_facing_box], base_only, [BOX], FENCE)["footprint_hits"] == 0
    assert T.check_poses([pose_facing_box], boxes, [BOX], FENCE)["footprint_hits"] == 1
    turned_away = (0.1, 0.0, math.pi)  # same place, arm pointing the other way
    assert T.check_poses([turned_away], boxes, [BOX], FENCE)["footprint_hits"] == 0


def test_the_checker_catches_leaving_the_fence():
    out = T.check_poses([(0.0, 0.0, 0.0), (1.1, 0.0, 0.0)], T.footprint_boxes((0.1, 0.1)), [], FENCE)
    assert out["outside_fence"] == 1 and out["footprint_hits"] == 0


def test_convex_overlap():
    a = box_polygon((0, 0), (1, 1))
    assert T.convex_overlap(a, box_polygon((0.6, 0), (0.4, 0.4)))  # 0.4 .. 0.8 against -0.5 .. 0.5
    assert not T.convex_overlap(a, box_polygon((0.9, 0), (0.4, 0.4)))  # 0.7 .. 1.1: apart
    assert T.convex_overlap(a, box_polygon((0.6, 0.6), (0.4, 0.4), 45))  # its corner (x + y = 0.92) pokes past (0.5, 0.5)
    assert not T.convex_overlap(a, box_polygon((0.75, 0.75), (0.4, 0.4), 45))  # x + y = 1.22: apart
    assert not T.convex_overlap(a, box_polygon((1.2, 1.2), (0.4, 0.4), 45))  # corner to corner, apart


def test_a_short_session_goes_around_obstacles_without_violations(tmp_path):
    with SimRobot(start=(0.3, 0.3, math.pi)) as sim:
        env = RealEnv(sim.url, np.eye(4))
        try:
            import time

            t0 = time.monotonic()
            while not env.localization().live and time.monotonic() - t0 < 5:
                time.sleep(0.02)
            res = T.run_session(env, half=1.0, trials=6, seed=3, obstacles=2, box=0.3,
                                boxes=T.footprint_boxes((0.56, 0.50), [(0.3, 0.2, 0.4)]), margin=0.1, min_move=0.1,
                                max_turn_deg=90, speed=0.5, yaw_speed=1.0, out=tmp_path, confirm=lambda _: None,
                                stop_on_violation=False, detour=1.0)  # fmt: skip
        finally:
            env.close()
    s = res["summary"]
    assert s["trials"] == 6 and s["reached"] == 6 and s["violations"] == 0
    assert s["detours"] >= 3 and s["segments_mean"] > 1.0  # A* actually routed around something
    assert (tmp_path / "map.txt").exists() and (tmp_path / "summary.json").exists()
