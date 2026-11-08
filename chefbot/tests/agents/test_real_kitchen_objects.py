"""go_to by an object's name on the real kitchen's object map (scripts/real_kitchen.py: explore, look, go_to_object)
and the live page's feed (scripts/live_view.py), on the fake robot of test_real_kitchen: the pole camera sees a
tomato near the island's south edge from above, whatever the base does."""

import json
import math

import numpy as np
import pytest

from chefbot.agent.world.map import WorldMap
from chefbot.interface.types import Pose2D
from chefbot.scripts.live_view import LiveFeed, map_json
from chefbot.scripts.real_kitchen import KitchenOptions, RealKitchen
from chefbot.tests.agents.test_real_kitchen import MAP, FakeKitchen, Host
from chefbot.tests.perception.test_kitchen_map import K, T_down, frame

TOMATO = (1.10, -2.45)  # 0.12 m in from the island's south edge


class Frame:
    def __init__(self, rgb, depth, K, T):
        self.rgb, self.depth, self.intrinsics, self.cam_to_world = rgb, depth, K, T


class SeeingKitchen(FakeKitchen):
    """The pole camera straight above the island, wherever the base is (its pose given in odom, as the env does)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.kitchen = None
        self.frames = 0

    def get_camera_rgbd(self, name):
        self.frames += 1
        W = self.kitchen._world_from_odom() @ self.kitchen._arm_offset()
        rgb, depth, _ = frame(TOMATO)
        return Frame(rgb, depth, K, np.linalg.inv(W) @ T_down())


class Yoloe:
    def __init__(self):
        self.calls = []

    def __call__(self, op, image, **args):
        self.calls.append((op, args.get("model")))
        return {"detections": frame(TOMATO)[2]}


def kitchen(tmp_path, live=True, **opts):
    env = SeeingKitchen(mocap=Pose2D(1.26, -2.94, math.radians(87)))
    feed = LiveFeed(tmp_path / "live", env, period_s=0.05) if live else None
    k = RealKitchen(Host(env), MAP, Yoloe(), tmp_path / "run", options=KitchenOptions(naming="off", **opts),
                    world_map=WorldMap.from_dict({}), live=feed)  # fmt: skip
    env.kitchen = k
    return k


def test_the_sweeps_stops_all_have_a_stance(tmp_path):
    k = kitchen(tmp_path, live=False)
    for route in ("full", "south", "stove"):
        stops = k.sweep_stops(route)
        assert stops and all("stance" in s for s in stops), [s.get("error") for s in stops]
    south = [s for s in k.sweep_stops("south") if s["side"] == "south"]
    assert all(80 < s["stance"][2] < 95 for s in south)  # facing the island from its south side (+y)


def test_explore_then_go_to_the_tomato(tmp_path):
    k = kitchen(tmp_path)
    r = k.explore("stove")  # a dry run: the drives planned, not driven; a look at each of the 2 stops
    assert r["ok"] and r["objects"] == ["tomato"] and k.env.frames == 2
    assert k.vision.calls == [("detect", "yoloe")] * 2
    assert k.world_map.get("tomato").inside == "island" and k.world_map.get("tomato").kind == "object"
    saved = json.loads((tmp_path / "run" / "objects.json").read_text())
    assert saved["objects"][0]["xyz"][:2] == pytest.approx(TOMATO, abs=0.01)

    g = k.go_to("tomato")
    assert g["ok"] and g["object"] == "tomato" and g["side"] == "south" and g["within_rule"]
    sx, sy, syaw = g["stance"]
    assert sx == pytest.approx(TOMATO[0], abs=0.05) and sy < -2.9 and 80 < syaw < 95  # in front of it, facing it
    assert "no object 'moon'" in k.go_to("moon")["reason"]
    assert k.go_to("island_north")["ok"]  # the map's places as before

    k.live.close()
    state = json.loads((tmp_path / "live" / "state.json").read_text())
    assert state["robot"]["live"] and state["objects"][0]["name"] == "tomato" and state["target"]["name"] == "island_north"
    assert (tmp_path / "live" / "pole.jpg").exists()
    whats = {t["what"] for t in state["timings"]}
    assert {"drive", "frame", "detect", "map"} <= whats
    rows = [json.loads(l) for l in (tmp_path / "run" / "timings.jsonl").read_text().splitlines()]
    assert sum(r["what"] == "detect" for r in rows) == 2


def test_frames_while_driving_go_into_the_map_too(tmp_path, monkeypatch):
    k = kitchen(tmp_path, live=False, capture="drive", capture_hz=50.0)
    real_drive = RealKitchen.drive

    def slow_drive(self, goal, approach_m=0.0, label="goal"):
        import time

        time.sleep(0.15)  # long enough for a few frames at 50 Hz
        return real_drive(self, goal, approach_m, label)

    monkeypatch.setattr(RealKitchen, "drive", slow_drive)
    r = k.explore("stove")
    assert all(s.get("drive_frames", 0) >= 2 for s in r["stops"]) and "look" not in r["stops"][0]
    assert r["objects"] == ["tomato"]


def test_the_live_pages_map():
    m = map_json(MAP)
    assert {"island", "stove_counter"} <= {b["fixture"] for b in m["boxes"]} and "island_south" in m["places"]
