"""The real_kitchen agent (the tomato demo): its registry and session, the host-v1 skills' kitchen
(scripts/real_kitchen.py) on a fake robot (find placing what a camera sees in the map's frame by mocap, every place
reaching every other), and a dry run of the whole job through run_agent."""

import math
from pathlib import Path

import numpy as np
import pytest

from chefbot import agents
from chefbot.agents.session import Session
from chefbot.interface.types import Pose2D
from chefbot.interface.workspace import READY
from chefbot.scripts import run_agent
from chefbot.scripts.real_kitchen import RealKitchen
from chefbot.tests.agents.test_real_arm import Scripted
from chefbot.tests.onboard.test_real_host import FakeRobot

REPO = Path(__file__).resolve().parents[3]
MAP = REPO / "maps/src_kitchen_work"
JOB = REPO / "chefbot/jobs/tomato_to_bowl.yaml"


class Loc:
    def __init__(self, mocap, odom):
        self.mocap_pose, self.odom_pose, self.live = mocap, odom, True


class Frame:
    def __init__(self, rgb, depth, K, T):
        self.rgb, self.depth, self.intrinsics, self.cam_to_world = rgb, depth, K, T


class FakeKitchen(FakeRobot):
    """FakeRobot with mocap (the base at `mocap` in the world, odometry at the origin) and one camera looking straight
    down from 1 m over the odom point (0.5, 0)."""

    require_arm = True
    server_limits: dict = {}

    def __init__(self, mocap=Pose2D(1.0, 2.0, math.pi / 2)):
        super().__init__()
        self.q = np.array(READY)  # the ready position: the arm inside the plate's outline, as the base drives
        self.mocap = mocap
        self.camera_mounts = {"pole": {"parent": "base"}}

    def localization(self):
        return Loc(self.mocap, Pose2D(0.0, 0.0, 0.0))

    def raw_state(self):
        return {"arm_q": list(self.q), "arm_live": True, "base_driver_live": True}

    def get_camera_rgbd(self, name):
        K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking down
        T[:3, 3] = [0.5, 0.0, 1.0]
        return Frame(np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32), K, T)


class FakeVision:
    def __call__(self, op, image, **args):
        if op == "detect":
            return {"detections": [{"label": args["queries"][0], "score": 0.8, "box": [310, 230, 330, 250]}]}
        mask = np.zeros((1, 480, 640), bool)
        mask[0, 230:250, 310:330] = True
        path = Path(image).with_suffix(".mask.npy")
        np.save(path, mask)
        return {"masks": str(path)}


class Host:
    def __init__(self, env, dry_run=True):
        self.raw_env, self.dry_run, self.calls = env, dry_run, []

    def halt(self):
        self.calls.append("halt")

    def start_here(self):
        self.calls.append("start_here")


def test_its_registry_and_session():
    s = Session.open(agents.load("real_kitchen"), env="real")
    assert [(x.name, x.path) for x in s.skills] == [
        ("go_to", "go_to/host_v1"), ("explore", "explore/host_v1"), ("find", "find/host_v1"), ("ready", "ready/host_v1"),
        ("pick_up", "kitchen_use/run1_v155_fr3"), ("place_in", "place_in/v1")]  # fmt: skip
    pick = next(x for x in s.skills if x.name == "pick_up").schema["function"]
    assert pick["parameters"]["required"] == ["object"]  # use: hold and the rest by its registry's defaults
    assert all(x.description for x in s.skills)


def test_find_puts_what_the_camera_sees_in_the_maps_frame(tmp_path):
    k = RealKitchen(Host(FakeKitchen(mocap=Pose2D(1.0, 1.85, math.pi / 2))), MAP, FakeVision(), tmp_path)
    r = k.find("tomato")
    # the odom point (0.5, 0), the arm 0.1825 m further ahead than the env puts it (and its plate 0.105 m higher): (0.6825,
    # 0); with the base at (1, 1.85) facing +y in the world, (1, 2.5325); the top at the depth plane, z 0.105
    assert r["found"] and r["world"] == pytest.approx([1.0, 2.5325, 0.105], abs=0.01)
    assert r["on"] == "counter"  # the test counter (Chef-botcounter-top) is at (1.49, 2.06), 2.26 x 0.96
    assert k.found["tomato"] is r


def test_go_to_knows_the_maps_places(tmp_path):
    k = RealKitchen(Host(FakeKitchen()), MAP, FakeVision(), tmp_path)
    assert k.go_to("island_north")["ok"]
    assert "no place 'moon'" in k.go_to("moon")["reason"]


def test_a_dry_run_of_the_job(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_URL", "http://127.0.0.1:1")  # the skills' check: never called (find is stubbed)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    tomato = {"found": True, "object": "tomato", "world": [1.0, -2.52, 0.95], "on": "island"}  # by its south edge
    monkeypatch.setattr(RealKitchen, "find", lambda self, object: self.found.setdefault(object, tomato))
    calls = [("set_status", {"step_id": "s1", "status": "active"}), ("ready", {}), ("go_to", {"place": "island_south"}),
             ("find", {"object": "tomato"}),
             ("finish", {"summary": "stopped before the pick", "abandon": True})]  # fmt: skip
    llm, fake = Scripted(calls), FakeKitchen(mocap=Pose2D(1.26, -2.94, math.radians(87)))  # at island_south
    result = run_agent.run(JOB, 0, tmp_path / "runs", llm=llm, operator=lambda r: "abort", quiet=True, env="real",
                           agent="real_kitchen", env_options={"env": fake, "confirm": False, "dry_run": True})  # fmt: skip
    assert {"go_to", "find", "ready", "pick_up", "place_in"} <= set(llm.tools) and "approach" not in llm.tools
    oks = [r["success"] for r in llm.results]
    assert oks[1:4] == [True, True, True], llm.results  # ready, go_to, find
    assert "island" in llm.results[3]["message"] and fake.sent["base"] == 0 and fake.sent["arm"] == 0
    assert Path(result["run_dir"]).exists()
    import json

    first = json.loads((Path(result["run_dir"]) / "requests" / "0001.json").read_text())
    state = next(m["content"] for m in reversed(first["messages"]) if m["role"] == "user")
    assert "TASK STATE" in state and "PERCEPTION" not in state and "KITCHEN" not in state  # no detect, no map here


def test_every_place_of_the_work_map_reaches_every_other(tmp_path):
    # The base turns where the path has room (scripts/real_kitchen.py, real_nav go --map): from beside a counter a short
    # path can't turn round as it starts (counter_north to island_south, 180 deg in 0.9 m, failed turning evenly).
    from chefbot.robot.nav.heading_plan import escape_start, plan_at_headings

    k = RealKitchen(Host(FakeKitchen()), MAP, FakeVision(), tmp_path)
    names = [n for n in sorted(k.places) if not k.places[n].get("relative_to")]
    pose = {n: Pose2D(k.places[n]["x"], k.places[n]["y"], math.radians(k.places[n]["yaw_deg"])) for n in names}
    failed = []
    for a in names:
        k.env.mocap = pose[a]
        robot = k._robot()
        body, plan = robot._space, robot._space.with_margin(robot._cfg.safety_margin)
        for b in names:
            if a == b:
                continue
            g, m = pose[b], k.places[b].get("approach_m", 0.0)
            pre = Pose2D(g.x - m * math.cos(g.yaw), g.y - m * math.sin(g.yaw), g.yaw)
            begin = escape_start(plan.at(pose[a].yaw), body.at(pose[a].yaw), (pose[a].x, pose[a].y))
            goal_ok = (m > 0 and body.is_free(g.x, g.y, g.yaw)) or plan.is_free(g.x, g.y, g.yaw)  # as _run_go_to_pose
            if not goal_ok or plan_at_headings(plan, body, pose[a], begin, pre, g.yaw, style=robot._style())[0] is None:
                failed.append((a, b))
    assert len(names) == 5 and failed == []


def test_what_is_held_follows_the_real_gripper_after_each_skill():
    # The pick and the place are episodes the in-process tools never see: TASK STATE's "holding" comes from the gripper.
    import inspect
    from types import SimpleNamespace

    from chefbot.scripts.run_agent import _tracking_held

    gripper, nav = {"holds": False}, SimpleNamespace(held=None)
    held = {"nav": nav}

    def pick(*, object):
        gripper["holds"] = True
        return "picked"

    def place(*, target, object="it"):
        gripper["holds"] = False
        return "placed"

    pick.__signature__ = inspect.signature(pick)
    p, q = _tracking_held(pick, lambda: gripper["holds"], held), _tracking_held(place, lambda: gripper["holds"], held)
    assert inspect.signature(p) == inspect.signature(pick)  # the registry binds the call's arguments to it
    assert p(object="tomato") == "picked" and nav.held == {"id": "tomato"}
    assert p(object="bowl") == "picked" and nav.held == {"id": "tomato"}  # still the tomato: nothing let go of it
    assert q(target="bowl", object="tomato") == "placed" and nav.held is None
