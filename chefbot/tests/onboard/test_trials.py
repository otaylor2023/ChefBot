"""scripts/real_trials: footprint geometry, the violation checker, and a short session against onboard/sim.py."""

import json
import math

import numpy as np
import pytest

from chefbot.environments.real.env import RealEnv
from chefbot.interface.types import Pose2D
from chefbot.interface.workspace import Footprint
from chefbot.onboard.sim import SimRobot
from chefbot.scripts import real_trials as T
from chefbot.environments.real.build_map import box_polygon

FENCE = box_polygon((0.0, 0.0), (2.0, 2.0))
BOX = box_polygon((0.6, 0.0), (0.2, 0.2))  # 0.5 .. 0.7 in x


ARMED = Footprint((-0.2, 0.2, -0.15, 0.15), np.array([[0.4, 0.0, 0.1]]))  # a 0.4 x 0.3 base, an arm disc 0.4 m ahead
BASE_ONLY = Footprint((-0.2, 0.2, -0.15, 0.15), np.zeros((0, 3)))


def test_the_footprint_turns_with_the_base():
    assert T.outline(ARMED, (0.0, 0.0, 0.0))[-1] == pytest.approx((0.4, 0.0))
    assert T.outline(ARMED, (0.0, 0.0, math.pi / 2))[-1] == pytest.approx((0.0, 0.4))


def test_the_checker_catches_the_arm_reaching_an_obstacle_the_base_clears():
    pose_facing_box = (0.1, 0.0, 0.0)  # base front at 0.3, arm disc 0.4 .. 0.6: into the box (0.5 .. 0.7)
    assert T.check_poses([pose_facing_box], BASE_ONLY, [BOX], FENCE)["footprint_hits"] == 0
    assert T.check_poses([pose_facing_box], ARMED, [BOX], FENCE)["footprint_hits"] == 1
    turned_away = (0.1, 0.0, math.pi)  # same place, arm pointing the other way
    out = T.check_poses([turned_away], ARMED, [BOX], FENCE)
    assert out["footprint_hits"] == 0 and out["min_clearance_m"] == pytest.approx(0.5 - 0.3, abs=0.006)


def test_the_checker_catches_leaving_the_fence():
    out = T.check_poses([(0.0, 0.0, 0.0), (1.1, 0.0, 0.0)], T.scaled(BASE_ONLY, 0.25), [], FENCE)
    assert out["outside_fence"] == 1 and out["footprint_hits"] == 0


def test_the_checker_catches_an_obstacle_inside_the_footprint():
    small = box_polygon((0.0, 0.0), (0.05, 0.05))  # wholly under the plate: no corner of the plate inside it
    assert T.check_poses([(0.0, 0.0, 0.3)], BASE_ONLY, [small], FENCE)["footprint_hits"] == 1


def test_a_short_session_goes_around_obstacles_without_violations(tmp_path):
    with SimRobot(start=(0.3, 0.3, math.pi)) as sim:
        env = RealEnv(sim.url, np.eye(4))
        try:
            import time

            t0 = time.monotonic()
            while not env.localization().live and time.monotonic() - t0 < 5:
                time.sleep(0.02)
            res = T.run_session(env, half=1.0, trials=6, seed=3, obstacles=2, box=0.3,
                                footprint=Footprint((-0.28, 0.28, -0.24, 0.24), np.array([[0.4, 0.0, 0.1]])), margin=0.1, min_move=0.1,
                                max_turn_deg=90, speed=0.5, yaw_speed=1.0, out=tmp_path, confirm=lambda _: None,
                                stop_on_violation=False, detour=1.0)  # fmt: skip
        finally:
            env.close()
    s = res["summary"]
    assert s["trials"] == 6 and s["reached"] == 6 and s["violations"] == 0
    assert s["detours"] >= 3 and s["segments_mean"] > 1.0  # A* actually routed around something
    assert (tmp_path / "map.txt").exists() and (tmp_path / "summary.json").exists()


def test_long_segments_are_split_into_steps_go_accepts():
    from chefbot.scripts.real_nav import subdivide

    pts = subdivide([(0.0, 0.0), (2.5, 0.0), (2.5, 0.4)], 1.0)
    assert pts == [(0.0, 0.0), pytest.approx((2.5 / 3, 0.0)), pytest.approx((5 / 3, 0.0)), (2.5, 0.0), (2.5, 0.4)]
    assert max(math.dist(a, b) for a, b in zip(pts, pts[1:])) <= 1.0


def test_stations_are_square_to_the_sides_even_of_a_rotated_counter():
    from chefbot.environments.real.build_map import stations

    spec = {"obstacles": [{"name": "c", "box": {"center": [1.0, 2.0], "size": [2.0, 1.0], "yaw_deg": 30.0}}],
            "stations": [{"obstacle": "c", "prefix": "c", "standoff": 0.4, "approach": 0.25}]}  # fmt: skip
    st = stations(spec)
    assert set(st) == {"c_east", "c_west", "c_north", "c_south"}
    for p in st.values():
        yaw = math.radians(p["yaw_deg"])
        to_centre = math.atan2(2.0 - p["y"], 1.0 - p["x"])
        assert abs(math.atan2(math.sin(to_centre - yaw), math.cos(to_centre - yaw))) < 1e-6  # facing the middle
        assert (math.degrees(yaw) - 30.0) % 90 == pytest.approx(0.0, abs=1e-6) or (math.degrees(yaw) - 30.0) % 90 == pytest.approx(90.0, abs=1e-6)
    e = st["c_east"]  # along the long axis: half length 1.0 + 0.4 out
    assert math.hypot(e["x"] - 1.0, e["y"] - 2.0) == pytest.approx(1.4) and e["approach_m"] == 0.25


def _sim_env():
    import time

    sim = SimRobot(start=(0.3, 0.3, math.pi))
    env = RealEnv(sim.url, np.eye(4))
    t0 = time.monotonic()
    while not (env.localization().live and env.raw_state().get("base_driver_live")) and time.monotonic() - t0 < 5:
        time.sleep(0.02)
    return sim, env


def test_smooth_following_goes_around_an_obstacle_in_one_motion(tmp_path):
    from chefbot.scripts.real_map import build
    from chefbot.scripts.real_nav import go_planned

    sim, env = _sim_env()
    try:
        s = env.localization().mocap_pose
        grid, _ = build({"resolution": 0.05, "bounds": [[s.x - 2.1, s.y - 1.6], [s.x + 1.6, s.y + 1.6]],
                         "obstacles": [{"name": "box", "box": {"center": [s.x - 0.6, s.y], "size": [0.3, 0.3]}}]})  # fmt: skip
        r = go_planned(env, Pose2D(s.x - 1.4, s.y, s.yaw), grid, confirm=lambda _: None, log_dir=tmp_path, speed=0.5,
                       yaw_speed=1.0, smooth=True)  # fmt: skip
    finally:
        env.close()
        sim.close()
    assert r["ok"], r["reason"]
    assert len(r["legs"]) == 1 and len(r["path"]) > 2  # one continuous follow around it, not a stop per corner
    rows = [json.loads(l) for l in (tmp_path / "leg00.jsonl").read_text().splitlines()]
    drive = [x for x in rows if x["phase"] == "drive"]
    assert all("carrot" in x for x in drive) and max(x["off_line"] for x in drive) < 0.15
    from chefbot.interface.workspace import BODY
    from chefbot.robot.nav.footprint import FootprintSpace

    body = FootprintSpace(grid, BODY.footprint())  # the plate, turned with the base
    assert all(body.is_free(*x["pose"]) for x in rows)


def test_a_station_is_reached_square_after_a_straight_approach(tmp_path):
    from chefbot.scripts.real_map import build
    from chefbot.scripts.real_nav import go_planned

    sim, env = _sim_env()
    try:
        s = env.localization().mocap_pose
        grid, _ = build({"resolution": 0.05, "bounds": [[s.x - 2.0, s.y - 2.0], [s.x + 2.0, s.y + 2.0]]})
        goal = Pose2D(s.x + 0.8, s.y + 0.5, math.pi / 2)  # facing +y, as at a counter's south side
        r = go_planned(env, goal, grid, confirm=lambda _: None, log_dir=tmp_path, speed=0.5, yaw_speed=1.0, approach_m=0.25)
    finally:
        env.close()
        sim.close()
    assert r["ok"] and abs(r["final_yaw_error_deg"]) < 1.0
    assert len(r["legs"]) == 2  # follow to the approach start, then straight in
    rows = [json.loads(l)["pose"] for l in (tmp_path / "leg01.jsonl").read_text().splitlines()]
    last = [p for p in rows if goal.y - 0.2 <= p[1] <= goal.y - 0.02]  # the last ~0.2 m in
    assert last and max(abs(p[0] - goal.x) for p in last) < 0.02  # straight along the normal
    assert max(abs(math.degrees(math.atan2(math.sin(p[2] - goal.yaw), math.cos(p[2] - goal.yaw)))) for p in last) < 3.0
