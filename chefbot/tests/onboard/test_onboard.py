"""The onboard server against a fake Redis holding what the drivers publish, and RealEnv against it."""

import json
import math
import threading
import time

import numpy as np
import pytest

from chefbot.environments.real.env import RealEnv, RobotHalted
from chefbot.interface.types import Pose2D
from chefbot.onboard import keys as K
from chefbot.onboard.frames import BaseFrames, compose
from chefbot.onboard.mocap import MocapFreshness
from chefbot.onboard.server import build

Q = [0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785]


from chefbot.onboard.sim import FakeRedis, SimBase  # noqa: E402  (the shared stand-ins)


def new_frame(redis: FakeRedis) -> None:
    """A tick of the live drivers: the mocap publisher rewrites the pose (here the same numbers, written anew)
    and the base driver counts its heartbeat up."""
    raw = redis.values[K.MOCAP_POSITION]
    redis.values[K.MOCAP_POSITION] = raw[:-1] if raw.endswith(" ") else raw + " "
    if K.BASE_HEARTBEAT in redis.values:
        redis.values[K.BASE_HEARTBEAT] = str(int(redis.values[K.BASE_HEARTBEAT]) + 1)


def live_state(robot, redis: FakeRedis) -> dict:
    """A state read with the publisher running: a frame lands between two reads."""
    robot.read_state()
    new_frame(redis)
    return robot.read_state()


def driver_values(odom=(1.0, 2.0, 0.5), mocap_xy=(3.0, 4.0), mocap_yaw=1.0, valid=True) -> dict:
    half = mocap_yaw / 2
    return {
        K.JOINT_POSITIONS: json.dumps(Q),
        K.JOINT_VELOCITIES: json.dumps([0.0] * 7),
        K.GRIPPER_CURRENT_WIDTH: "0.0791",
        K.GRIPPER_MAX_WIDTH: "0.08",
        K.GRIPPER_READY: "1",
        K.BASE_POSE: f"[{odom[0]}, {odom[1]}, {odom[2]}]",
        K.BASE_HEARTBEAT: "1",
        K.BASE_VELOCITY: "[0.0, 0.0, 0.0]",
        K.MOCAP_POSITION: f"[{mocap_xy[0]}, {mocap_xy[1]}, 0.3]",
        K.MOCAP_ORIENTATION: f"[0.0, 0.0, {math.sin(half)}, {math.cos(half)}]",  # x, y, z, w
        K.MOCAP_VALID: "true" if valid else "false",
        K.ACTIVE_CONTROLLER: "joint_controller",
        K.CARTESIAN_CURRENT_POSITION: "[0.4, 0.0, 0.5]",
        K.CARTESIAN_CURRENT_ORIENTATION: json.dumps(np.eye(3).tolist()),
    }


CONFIG = {
    "server": {"host": "127.0.0.1", "port": 0, "state_hz": 100, "watchdog_s": 0.3, "camera_idle_s": 10},
    "base": {"driver_max_vel_xy": 0.125, "driver_max_vel_yaw": 0.4, "marker_yaw_offset_deg": 0.0, "marker_xy": [0.0, 0.0]},
    "gripper": {"speed": 0.05, "force": 20.0},
    "limits": {"base_max_vel_xy": 0.125, "base_max_vel_yaw": 0.4},
    "cameras": [],
}


# -- frames -------------------------------------------------------------------------


def test_marker_offsets_give_the_base_pose():
    frames = BaseFrames(marker_yaw_offset=math.radians(-38.17), marker_xy=(0.1, 0.0))
    half = math.radians(90.0 - 38.17) / 2  # the marker reads 51.83 deg: the base faces 90
    x, y, yaw = frames.base_from_mocap([1.0, 1.0, 0.3], [0.0, 0.0, math.sin(half), math.cos(half)])
    assert yaw == pytest.approx(math.pi / 2)
    assert (x, y) == pytest.approx((1.0, 0.9))  # the marker is 0.1 m ahead of the center, which faces +y


def test_world_and_odometry_round_trip_through_the_tie():
    frames = BaseFrames()
    frames.tie([3.0, 4.0, 1.0], [1.0, 2.0, 0.5])
    np.testing.assert_allclose(frames.to_world([1.0, 2.0, 0.5]), [3.0, 4.0, 1.0], atol=1e-12)
    goal = np.array([5.0, -1.0, 2.0])
    np.testing.assert_allclose(frames.to_world(frames.to_odom(goal)), goal, atol=1e-12)


# -- robot_io -----------------------------------------------------------------------


def test_state_is_in_the_mocap_world_and_carries_on_through_odometry():
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    s = live_state(robot, redis)
    np.testing.assert_allclose(s["base_pose"], [3.0, 4.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(s["mocap_pose"], [3.0, 4.0, 1.0], atol=1e-9)
    assert s["localized"] and s["mocap_valid"] and s["mocap_fresh"]
    np.testing.assert_allclose(s["arm_q"], Q)
    assert s["gripper_width"] == pytest.approx(0.0791)
    # Mocap lost; the wheels move the base 0.5 m forward along its odometry heading.
    moved = compose([1.0, 2.0, 0.5], [0.5, 0.0, 0.0])
    redis.values.update({K.MOCAP_VALID: "false", K.BASE_POSE: json.dumps(moved.tolist())})
    s = robot.read_state()
    assert not s["mocap_valid"] and s["localized"]
    np.testing.assert_allclose(s["base_pose"], compose([3.0, 4.0, 1.0], [0.5, 0.0, 0.0]), atol=1e-9)


def beat(redis: FakeRedis) -> None:
    """The base driver alone ticks (mocap unchanged)."""
    redis.values[K.BASE_HEARTBEAT] = str(int(redis.values[K.BASE_HEARTBEAT]) + 1)


def test_without_mocap_ever_the_world_is_odometry():
    redis = FakeRedis(driver_values(valid=False))
    robot = build(CONFIG, redis).robot
    robot.read_state()
    beat(redis)
    s = robot.read_state()
    assert not s["localized"] and s["base_driver_live"]
    np.testing.assert_allclose(s["base_pose"], [1.0, 2.0, 0.5])


def test_a_pose_left_in_redis_by_a_dead_publisher_is_never_trusted():
    # Redis keeps the publisher's last values, tracking_valid = 1 included, after it stops (even across reboots).
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    for _ in range(3):
        s = robot.read_state()
        assert s["mocap_valid"] and not s["mocap_fresh"] and s["mocap_age_s"] is None
        assert s["mocap_pose"] is None and not s["localized"]
        beat(redis)  # the base driver is running
    np.testing.assert_allclose(s["base_pose"], [1.0, 2.0, 0.5])  # plain odometry


def test_a_publisher_that_stops_goes_stale_and_the_world_carries_on_through_odometry():
    now = [0.0]
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    robot.mocap = MocapFreshness(stale_after_s=0.25, clock=lambda: now[0])
    robot.read_state()
    new_frame(redis)
    now[0] = 0.1
    s = robot.read_state()
    assert s["mocap_fresh"] and s["mocap_age_s"] == pytest.approx(0.0) and s["localized"]
    # The publisher dies here. Within stale_after_s it cannot be told from a still robot: the frozen
    # pose still counts (the residual window, at most driver_max_vel_xy * stale_after_s of error).
    now[0] = 0.3
    s = robot.read_state()
    assert s["mocap_fresh"] and s["mocap_age_s"] == pytest.approx(0.2)
    now[0] = 0.4
    s = robot.read_state()
    assert not s["mocap_fresh"] and s["mocap_valid"] and s["mocap_pose"] is None
    # From then on the wheels move the base 0.5 m forward and the world pose follows them, not the frozen mocap.
    moved = compose([1.0, 2.0, 0.5], [0.5, 0.0, 0.0])
    redis.values[K.BASE_POSE] = json.dumps(moved.tolist())
    now[0] = 1.0
    s = robot.read_state()
    assert s["localized"] and not s["mocap_fresh"]
    np.testing.assert_allclose(s["base_pose"], compose([3.0, 4.0, 1.0], [0.5, 0.0, 0.0]), atol=1e-9)


def test_mocap_alone_gives_the_pose_without_the_base_driver():
    values = driver_values()
    values.pop(K.BASE_POSE)
    redis = FakeRedis(values)
    s = live_state(build(CONFIG, redis).robot, redis)
    assert s["base_pose"] is None and not s["localized"]
    np.testing.assert_allclose(s["mocap_pose"], [3.0, 4.0, 1.0], atol=1e-9)


def test_commands_are_written_in_the_drivers_formats_and_repeats_are_not():
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    robot.read_state()
    robot.arm_joints(Q)
    robot.arm_joints(Q)
    assert json.loads(redis.values[K.JOINT_GOAL_POSITION]) == Q
    assert redis.written(K.ACTIVE_CONTROLLER) == ["joint_controller"]  # switched once
    robot.arm_pose([0.4, 0.1, 0.5], np.eye(3))
    assert json.loads(redis.values[K.CARTESIAN_GOAL_POSITION]) == [0.4, 0.1, 0.5]
    assert json.loads(redis.values[K.CARTESIAN_GOAL_ORIENTATION]) == np.eye(3).tolist()
    assert redis.written(K.ACTIVE_CONTROLLER) == ["joint_controller", "cartesian_controller"]
    robot.gripper(True)
    robot.gripper(True)
    assert redis.written(K.GRIPPER_MODE) == ["g"]
    assert redis.values[K.GRIPPER_GRASP_SUCCESS] == "pending"
    robot.gripper(False)
    assert redis.written(K.GRIPPER_MODE) == ["g", "m"] and redis.values[K.GRIPPER_DESIRED_WIDTH] == "0.08"


def test_base_targets_go_to_the_driver_in_odometry_and_never_faster_than_it():
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    live_state(robot, redis)  # ties world (3, 4, 1) to odometry (1, 2, 0.5)
    robot.base_target([3.0, 4.0, 1.0], (1.0, 1.0, 1.0))
    odom = [float(v) for v in redis.values[K.BASE_DESIRED_POSE].strip("[]").split(",")]
    np.testing.assert_allclose(odom, [1.0, 2.0, 0.5], atol=1e-9)
    assert redis.values[K.BASE_MAX_VEL_SCALE] == "1.000"
    from chefbot.interface.limits import base_cap

    cap = base_cap("tidybot_fr3")["vel"]  # the driver runs at the robot's hard base cap
    robot.base_target([3.0, 4.0, 1.0], (cap[0] / 2, cap[1] / 2, cap[2] / 2))
    assert redis.values[K.BASE_MAX_VEL_SCALE] == "0.500"


def test_halt_holds_the_arm_where_it_is_and_stops_the_base():
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    robot.arm_pose([0.4, 0.1, 0.5], np.eye(3))
    robot.halt()
    assert json.loads(redis.values[K.JOINT_GOAL_POSITION]) == Q
    assert redis.values[K.ACTIVE_CONTROLLER] == "joint_controller"
    assert redis.values[K.BASE_STOP] == "stop"


# -- server and RealEnv ---------------------------------------------------------------


class FakeCameras:
    cameras = {"wrist": None}

    def grab(self, name):
        return {"rgb": np.zeros((4, 6, 3), np.uint8), "depth": np.ones((4, 6), np.float32), "intrinsics": np.eye(3)}

    def close_idle(self):
        pass

    def close(self):
        pass


class FakePublisher:
    """The mocap publisher: a new frame every few milliseconds until stopped."""

    def __init__(self, redis: FakeRedis, period_s: float = 0.005):
        self.stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(redis, period_s), daemon=True)
        self._thread.start()

    def _run(self, redis, period_s):
        while not self.stopped.wait(period_s):
            new_frame(redis)

    def stop(self):
        self.stopped.set()
        self._thread.join()


@pytest.fixture
def served():
    redis = FakeRedis(driver_values())
    publisher = FakePublisher(redis)
    server = build(CONFIG, redis)
    server.cameras = FakeCameras()
    server.start()
    control = np.eye(4)
    control[2, 3] = 0.1224
    env = RealEnv(server.url, control, cameras={"wrist": {"parent": "base", "T": np.eye(4)}})
    env.publisher = publisher
    yield server, env, redis
    publisher.stop()
    env.close()
    server.shutdown()


def wait_until(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_real_env_reads_the_robot_and_commands_it(served):
    server, env, redis = served
    wait_until(lambda: env.localization().localized)
    state = env.get_state()
    assert state.base_pose.x == pytest.approx(3.0) and state.base_pose.yaw == pytest.approx(1.0)
    np.testing.assert_allclose(state.arm_q, Q)
    # A flange pose goes out as the controller's point: 0.1224 m further along the flange's z.
    flange = np.eye(4)
    flange[:3, 3] = [0.4, 0.0, 0.3]
    assert env.set_arm_pose_target(flange)
    wait_until(lambda: K.CARTESIAN_GOAL_POSITION in redis.values)
    np.testing.assert_allclose(json.loads(redis.values[K.CARTESIAN_GOAL_POSITION]), [0.4, 0.0, 0.4224])
    assert env.set_base_target(Pose2D(3.0, 4.0, 1.0), (0.1, 0.1, 0.3))
    wait_until(lambda: K.BASE_DESIRED_POSE in redis.values)
    env.set_gripper(True)
    wait_until(lambda: redis.values.get(K.GRIPPER_MODE) == "g")
    rgbd = env.get_camera_rgbd("wrist")
    assert rgbd.rgb.shape == (4, 6, 3)
    np.testing.assert_allclose(rgbd.cam_to_world[:3, 3], [3.0, 4.0, 0.0])


def test_heartbeats_keep_an_idle_client_from_being_halted(served):
    server, env, _ = served
    env.set_arm_joint_targets(Q)
    time.sleep(0.8)
    assert server.halts == 0
    env.wait_for_next_tick(0.01)  # no halt to report


def test_a_silent_client_is_halted_and_told(served):
    server, env, redis = served
    env.set_arm_joint_targets(Q)
    env._closed.set()  # the heartbeat stops, as if this process hung
    wait_until(lambda: server.halts >= 1)
    assert redis.values[K.BASE_STOP] == "stop"
    wait_until(lambda: env.halted is not None)
    with pytest.raises(RobotHalted):
        env.wait_for_next_tick(0.01)


def test_disconnecting_after_commanding_halts_and_frees_the_server_for_the_next_client(served):
    server, env, _ = served
    wait_until(lambda: env.localization().localized)
    env.set_base_target(Pose2D(3.0, 4.0, 1.0), (0.1, 0.1, 0.3))
    wait_until(lambda: server.commanded)
    env.close()
    wait_until(lambda: server.halts >= 1)
    second = RealEnv(server.url, np.eye(4))
    second.get_state()
    second.close()


def test_a_client_that_only_watched_leaves_the_drivers_alone(served):
    server, env, redis = served
    wait_until(lambda: env.localization().live)
    loc = env.localization()
    assert loc.mocap_pose.x == pytest.approx(3.0) and loc.mocap_pose.yaw == pytest.approx(1.0)
    assert loc.mocap_fresh and loc.mocap_valid and loc.localized and loc.state_current
    time.sleep(0.4)  # longer than the watchdog: heartbeats only
    env.close()
    time.sleep(0.3)
    assert server.halts == 0
    assert [k for k, _ in redis.writes] == []  # not a single command key written
    second = RealEnv(server.url, np.eye(4))  # and the server is free again
    second.raw_state()
    second.close()


def test_localization_reports_a_dead_publisher(served):
    server, env, _ = served
    wait_until(lambda: env.localization().live)
    env.publisher.stop()
    wait_until(lambda: not env.localization().mocap_fresh, timeout_s=2.0)
    loc = env.localization()
    assert not loc.live and loc.mocap_pose is None and loc.mocap_valid  # the flag alone still says 1
    assert loc.mocap_age_s > 0.25


def test_a_second_client_is_refused(served):
    server, env, _ = served
    with pytest.raises(RuntimeError, match="another client"):
        RealEnv(server.url, np.eye(4))


def test_kept_cameras_stay_open_and_the_others_idle_out():
    from chefbot.onboard.cameras import Cameras

    class Cam:
        def __init__(self):
            self.open, self.last_used, self.opened = False, 0.0, 0

        def grab(self):
            self.open, self.opened = True, self.opened + 1
            return {}

        def close(self):
            self.open = False

    cams = Cameras([{"name": "wrist", "serial": "1"}, {"name": "pole", "serial": "2"}], idle_s=0.0)
    cams.cameras = {"wrist": Cam(), "pole": Cam()}
    cams.keep(["wrist"])
    assert cams.cameras["wrist"].open and cams.cameras["wrist"].opened == 1  # opened at once
    cams.cameras["pole"].grab()
    time.sleep(0.01)
    cams.close_idle()
    assert cams.cameras["wrist"].open and not cams.cameras["pole"].open
    cams.keep([])
    cams.close_idle()
    assert not cams.cameras["wrist"].open
    with pytest.raises(KeyError):
        cams.keep(["nope"])

# -- scripts/real_pose ----------------------------------------------------------------


def test_real_pose_prints_and_summarizes_what_the_robot_reports(served):
    from chefbot.scripts.real_pose import line, record, summarize

    server, env, _ = served
    wait_until(lambda: env.localization().live)
    samples = [record(env.localization(), 0.02 * i) for i in range(5)]
    assert line(env.localization()).startswith("LIVE")
    env.publisher.stop()
    wait_until(lambda: not env.localization().mocap_fresh)
    samples += [record(env.localization(), 0.1 + 0.02 * i) for i in range(5)]
    assert line(env.localization()).startswith("stale")
    summary = summarize(samples)
    assert "10 samples, 5 live (50%)" in summary
    assert "x 0.00 mm" in summary  # the fake publisher's pose does not move
    assert "longest stretch without a live pose: 0.10 s" in summary


# -- base driver liveness and the target guard ------------------------------------------


def test_odometry_left_in_redis_by_a_stopped_driver_is_not_used():
    # Seen on the robot (HT-0a): hb1::current_pose there with no base driver running.
    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    robot.read_state()
    new_frame(redis)
    redis.values[K.BASE_HEARTBEAT] = "1"  # the heartbeat never changes: no driver
    robot.read_state()
    new_frame(redis)
    redis.values[K.BASE_HEARTBEAT] = "1"
    s = robot.read_state()
    assert s["mocap_pose"] is not None and not s["base_driver_live"]
    assert s["base_pose"] is None and not s["localized"]  # no tie to the frozen odometry


def test_base_targets_need_live_mocap_and_a_live_driver():
    from chefbot.onboard.robot_io import NotLocalized

    redis = FakeRedis(driver_values())
    robot = build(CONFIG, redis).robot
    robot.read_state()  # nothing live yet
    with pytest.raises(NotLocalized):
        robot.base_target([3.0, 4.0, 1.0], (0.1, 0.1, 0.3))
    assert K.BASE_DESIRED_POSE not in redis.values and K.BASE_CMD_SEQ not in redis.values
    live_state(robot, redis)
    robot.base_target([3.0, 4.0, 1.0], (0.1, 0.1, 0.3))
    robot.base_target([3.0, 4.0, 1.0], (0.1, 0.1, 0.3))
    assert redis.values[K.BASE_CMD_SEQ] == "2"  # every target bumps the sequence the driver watches


def test_the_server_reports_a_refused_target_and_does_not_count_it_as_commanding(served):
    server, env, redis = served
    env.publisher.stop()
    redis.values[K.MOCAP_VALID] = "false"
    wait_until(lambda: not env.localization().mocap_valid)
    env.set_base_target(Pose2D(3.0, 4.0, 1.0), (0.1, 0.1, 0.3))
    wait_until(lambda: env.errors)
    assert "refused" in env.errors[-1] and not server.commanded
    assert K.BASE_CMD_SEQ not in redis.values


# -- scripts/real_nav: A to B against a simulated base ---------------------------------


@pytest.fixture
def sim():
    values = driver_values()
    values.pop(K.JOINT_POSITIONS)  # no arm driver
    redis = FakeRedis(values)
    base = SimBase(redis)
    server = build(CONFIG, redis)
    server.cameras = FakeCameras()
    server.start()
    env = RealEnv(server.url, np.eye(4))
    yield env, base, redis, server
    base.stop()
    env.close()
    server.shutdown()


def test_real_nav_goes_from_a_to_b_and_holds(sim, tmp_path):
    from chefbot.scripts.real_nav import go

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    start = env.localization().mocap_pose
    goal = Pose2D(start.x + 0.3, start.y - 0.2, start.yaw + 0.3)
    result = go(env, goal, speed=0.5, yaw_speed=1.0, confirm=lambda _: None, log_path=tmp_path / "go.jsonl")
    assert result["ok"] and result["reason"] == "reached", result
    assert result["final_error_m"] < 0.03 and abs(result["final_yaw_error_deg"]) < 3.0
    assert redis.values[K.BASE_STOP] in ("stop", "ok")  # held at the end
    assert len((tmp_path / "go.jsonl").read_text().splitlines()) > 3


def test_real_nav_stops_when_mocap_is_lost_mid_move(sim):
    from chefbot.scripts.real_nav import go

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    start = env.localization().mocap_pose
    goal = Pose2D(start.x + 1.0, start.y, start.yaw)
    threading.Timer(0.5, base.mocap_lost.set).start()
    result = go(env, goal, speed=0.2, yaw_speed=1.0, confirm=lambda _: None)
    assert not result["ok"] and "lost the live mocap pose" in result["reason"]
    wait_until(lambda: redis.values.get(K.BASE_STOP) == "ok" or redis.writes[-1][0] == K.BASE_STOP)
    odom_then = base.odom.copy()
    time.sleep(0.8)  # past the driver's timeout: the simulated base must not keep going
    assert np.linalg.norm(base.odom[:2] - odom_then[:2]) < 0.05


def test_real_nav_refuses_without_a_live_driver_or_too_far(sim):
    from chefbot.scripts.real_nav import go

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live)
    start = env.localization().mocap_pose
    far = go(env, Pose2D(start.x + 3.0, start.y, start.yaw), confirm=lambda _: None)
    assert not far["ok"] and not far["moved"] and "max-dist" in far["reason"]
    base.stop()  # the driver stops: no heartbeat
    wait_until(lambda: not env.raw_state().get("base_driver_live"))
    res = go(env, Pose2D(start.x + 0.2, start.y, start.yaw), confirm=lambda _: None)
    assert not res["ok"] and not res["moved"] and "base driver" in res["reason"]
    assert K.BASE_CMD_SEQ not in redis.values  # nothing was ever commanded


def test_real_nav_mark_averages_a_still_pose(sim, tmp_path):
    from chefbot.scripts.real_nav import load_points, mark

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live)
    p = mark(env, "A", path=tmp_path / "points.json", seconds=0.3)
    assert load_points(tmp_path / "points.json")["A"] == p and p["samples"] > 5
    assert (p["x"], p["y"]) == pytest.approx((2.0, 1.0), abs=1e-6)  # odometry at 0: the world offset


def test_relative_points_follow_the_origins_heading():
    from chefbot.scripts.real_nav import relative_point

    a = {"x": 1.0, "y": 2.0, "yaw_deg": 90.0}  # facing +y
    p = relative_point(a, forward=0.3, left=0.1, turn_deg=-15.0)
    assert (p["x"], p["y"], p["yaw_deg"]) == pytest.approx((0.9, 2.3, 75.0))


# -- scripts/real_characterize ----------------------------------------------------------


def test_characterize_plan_stays_inside_the_fence():
    from chefbot.scripts.real_characterize import plan

    moves = plan(half=0.2)
    assert len(moves) == 5 + 8 + 7 + 3 + 3
    assert all(max(abs(m["forward"]), abs(m["left"])) <= 0.2 for m in moves)
    with pytest.raises(ValueError):
        plan(half=0.1, step=0.3)


def test_the_fence_refuses_targets_outside_it_and_stops_a_move_that_leaves(sim):
    from chefbot.scripts.real_nav import Fence, go

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    c = env.localization().mocap_pose
    fence = Fence(c.x, c.y, math.degrees(c.yaw), half=0.1)
    out = go(env, Pose2D(c.x + 0.3, c.y, c.yaw), confirm=lambda _: None, fence=fence)
    assert not out["ok"] and not out["moved"] and "fence" in out["reason"]
    assert K.BASE_CMD_SEQ not in redis.values


def test_characterize_runs_a_session_and_reports(sim, tmp_path):
    from chefbot.scripts.real_characterize import plan, report, run

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    moves = [m for m in plan(half=0.2, speed=0.1) if m["name"] in ("forward-1", "forward-2", "left", "turn+45")]
    for m in moves:
        m["speed"] = 0.5  # the simulated base is quick
    results = run(env, 0.2, tmp_path, moves, yaw_speed=1.0, confirm=lambda _: None)
    assert len(results) == 8 and all(r["ok"] for r in results), [r["reason"] for r in results]
    text = report(tmp_path)
    assert "[repeat] 4 legs" in text and "[turns] 2 legs" in text and "centre drift in turns" in text
    assert "calibration hints" in text


# -- scripts/real_map and real_nav go --map: A* around obstacles -------------------------


def test_real_map_marks_bounds_and_obstacles():
    from chefbot.scripts.real_map import build

    grid, places = build({"resolution": 0.05, "bounds": [[0, 0], [2, 1]],
                          "obstacles": [{"name": "b", "box": {"center": [1.0, 0.5], "size": [0.4, 0.2], "yaw_deg": 0}}],
                          "places": {"p": {"x": 0.3, "y": 0.5, "yaw_deg": 90}}})  # fmt: skip
    assert grid.is_free(0.3, 0.5) and not grid.is_free(1.0, 0.5) and not grid.is_free(1.15, 0.55)
    assert not grid.is_free(-0.05, 0.5) and not grid.is_free(1.0, 1.05)  # outside the bounds
    assert grid.is_free(1.0, 0.75) and places["p"]["yaw_deg"] == 90


def test_real_nav_plans_around_an_obstacle_and_never_touches_it(sim, tmp_path):
    from chefbot.scripts.real_map import build
    from chefbot.scripts.real_nav import go_planned

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    s = env.localization().mocap_pose  # (2, 1, 0.5): the simulated base's world pose at odometry zero
    grid, _ = build({"resolution": 0.05, "bounds": [[s.x - 2.1, s.y - 1.6], [s.x + 1.6, s.y + 1.6]],
                     "obstacles": [{"name": "box", "box": {"center": [s.x - 0.6, s.y], "size": [0.3, 0.3]}}]})  # fmt: skip
    goal = Pose2D(s.x - 1.4, s.y, s.yaw)  # 0.25 m past the box's far edge grown by footprint + margin
    result = go_planned(env, goal, grid, footprint=0.33, margin=0.10, confirm=lambda _: None, log_dir=tmp_path,
                        speed=0.5, yaw_speed=1.0)  # fmt: skip
    assert result["ok"], result["reason"]
    assert len(result["path"]) > 2  # not the straight line through the box
    assert result["final_error_m"] < 0.03
    body = grid.inflate(0.33)
    for leg in sorted(tmp_path.glob("leg*.jsonl")):
        for line in leg.read_text().splitlines():
            x, y, _ = json.loads(line)["pose"]
            assert body.is_free(x, y)  # the footprint never overlapped the box


def test_real_nav_refuses_a_goal_inside_an_obstacle(sim):
    from chefbot.scripts.real_map import build
    from chefbot.scripts.real_nav import go_planned

    env, base, redis, server = sim
    wait_until(lambda: env.localization().live and env.raw_state().get("base_driver_live"))
    s = env.localization().mocap_pose
    grid, _ = build({"bounds": [[s.x - 1.6, s.y - 1.6], [s.x + 1.6, s.y + 1.6]],
                     "obstacles": [{"name": "box", "box": {"center": [s.x - 0.6, s.y], "size": [0.3, 0.3]}}]})  # fmt: skip
    r = go_planned(env, Pose2D(s.x - 0.6, s.y, s.yaw), grid, confirm=lambda _: None)
    assert not r["ok"] and not r["moved"] and "no path" in r["reason"]
    assert K.BASE_CMD_SEQ not in redis.values


def test_real_nav_stops_the_base_if_the_arm_moves_while_driving(sim):
    from chefbot.scripts.real_nav import go

    env, base, redis, server = sim
    q = np.array(Q, dtype=float)
    bump = threading.Event()
    stop = threading.Event()

    def arm_driver():  # live joint readings (sensor noise), and one joint moving 0.1 rad once bumped
        i = 0
        while not stop.wait(0.004):
            i += 1
            now = q + 1e-6 * (i % 7) + (np.r_[0.1, np.zeros(6)] if bump.is_set() else 0.0)
            redis.values[K.JOINT_POSITIONS] = json.dumps(now.tolist())

    t = threading.Thread(target=arm_driver, daemon=True)
    t.start()
    try:
        wait_until(lambda: env.localization().live and env.raw_state().get("arm_live") and env.raw_state().get("base_driver_live"))
        start = env.localization().mocap_pose
        threading.Timer(0.4, bump.set).start()
        r = go(env, Pose2D(start.x + 1.0, start.y, start.yaw), speed=0.2, yaw_speed=1.0, confirm=lambda _: None)
        assert not r["ok"] and "the arm moved" in r["reason"]
    finally:
        stop.set()
        t.join()
