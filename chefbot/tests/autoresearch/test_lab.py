"""The autoresearch lab without a simulator: the episode runner against a fake host (a real sandboxed skill process),
and the pick suites' splits."""

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from chefbot.autoresearch import config, models, setups, suites
from chefbot.autoresearch.episode import run_episode
from chefbot.interface.types import Pose2D


class FakeType:
    def skill_args(self, task):
        return {"object": "tomato"}

    def score(self, history, task):
        return {"success": bool(history[-1]["closed"])}


class FakeTask:
    name = "pick"
    type = FakeType()


class FakeHost:
    """Steps count as world time; the flange goes wherever it is told."""

    realtime = False

    def __init__(self):
        self.step_count, self.frames, self.flange, self.closed, self.modes = 0, [], np.eye(4), False, []

    def start(self, task, seed):
        return Pose2D(0.0, 0.0, 0.0)

    def step(self, row, mode="whole"):
        self.step_count += 1
        self.modes.append(mode)
        if mode == "arm":
            self.flange[:3, 3] = row[:3]
            self.closed = row[7] >= 0.5
        return {"pos_error_m": 0.0, "rot_error_rad": 0.0, "base_error_m": 0.0, "ik_ok": True}

    def proprio(self):
        return {"step": self.step_count, "flange_pose_odom": self.flange.tolist(), "base_pose_odom": [0.0, 0.0, 0.0],
                "gripper_closed_cmd": self.closed}  # fmt: skip

    def hold_row(self):
        return [0.0] * 7 + [0.0, 0.0, 0.0, 0.0]

    def rgbd(self):
        return {"wrist": {"rgb": np.zeros((4, 6, 3), np.uint8), "depth": np.ones((4, 6), np.float32), "K": np.eye(3),
                          "cam_to_odom": np.eye(4)}}  # fmt: skip

    def measure(self):
        return {"closed": self.closed}

    def debug(self):
        return {}


def episode(tmp_path: Path, code: str, access: dict | None = None) -> tuple[dict, FakeHost, str]:
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "main.py").write_text(textwrap.dedent(code))
    host = FakeHost()
    out = tmp_path / "out"
    access = access or {"llm": False, "vision": [], "arm_only": True}
    result = run_episode(host, FakeTask(), 0, skill, out, models.Ledger(tmp_path / "ledger.jsonl", 0.0), models.Vision(None), access,
                         record=False)  # fmt: skip
    return result, host, (out / "skill_log.txt").read_text()


def test_a_skill_gets_its_argument_moves_the_arm_and_is_scored(tmp_path):
    result, host, log = episode(tmp_path, """
        def run(robot, object):
            robot.log("asked for", object)
            robot.observe(["wrist"])
            r = robot.act_arm([[0.4, 0.0, 0.3, 1, 0, 0, 0, 1.0]] * 15)
            return {"ok": True, "z": r["proprio"]["flange_pose_root"][2][3]}
    """)
    assert "asked for tomato" in log
    assert result["success"] is True and result["returned"] == {"ok": True, "z": 0.3}
    assert result["steps"] == 15 and set(host.modes) == {"arm"} and result["end"] == "skill returned"


def test_arm_only_refuses_whole_body_and_base_moves(tmp_path):
    result, _, log = episode(tmp_path, """
        def run(robot, object):
            for call in (lambda: robot.act([[0] * 11]), lambda: robot.act_base([[0, 0, 0]])):
                try:
                    call()
                except RuntimeError as e:
                    robot.log("refused:", e)
    """)
    assert log.count("refused: this skill is arm only") == 2 and result["steps"] == 0


def test_the_world_runs_on_while_the_skill_thinks(tmp_path):
    result, host, _ = episode(tmp_path, """
        import time
        def run(robot, object):
            robot.proprio()
            time.sleep(1.0)
            robot.proprio()
    """)
    assert 13 <= result["steps"] <= 17 and set(host.modes) == {"whole"}  # 1 s at 15 Hz, holding the start pose
    assert result["think_s"] >= 1.0


def test_a_crash_is_reported_and_models_are_gated(tmp_path):
    result, _, log = episode(tmp_path, """
        import numpy as np
        def run(robot, object):
            try:
                robot.llm("gemini-3.1-flash-lite", "hi")
            except RuntimeError as e:
                robot.log("llm:", e)
            raise ValueError("boom")
    """)
    assert "llm: this skill may not call Gemini" in log
    assert result["end"] == "skill raised ValueError: boom" and "Traceback" in log


def test_the_sandbox_blocks_the_repo_and_the_network(tmp_path):
    repo_file = Path(__file__).resolve()
    result, _, log = episode(tmp_path, f"""
        def run(robot, object):
            for attempt in (lambda: open({str(repo_file)!r}).read(), lambda: __import__("socket").create_connection(("example.com", 80), 2)):
                try:
                    attempt()
                    robot.log("ALLOWED")
                except Exception as e:
                    robot.log("blocked", type(e).__name__)
    """)
    assert "ALLOWED" not in log and log.count("blocked PermissionError") == 2 and result["sandbox_blocked"] == 2


# -- the pick suites ---------------------------------------------------------------------


def pick_parts():
    return {part: [setups.load(it.setup) for it in s.items] for part, s in suites.load_set("robocasa_pick").items()}


def test_every_lab_names_suites_and_setups_that_exist():
    for path in config.LABS.glob("*.yaml"):
        cfg = config.load(str(path))
        for part in cfg.suites:
            for it in suites.load(cfg.suite(part)).items:
                assert it.task in setups.load(it.setup).task_names, it.id


def test_val_and_test_share_no_kitchen_or_object_category_with_dev():
    parts = pick_parts()

    def kitchens(part):
        return {(s.scene["layout"], s.scene["style"]) for s in parts[part]}

    def categories(part):
        return {o["category"] for s in parts[part] for o in s.scene["objects"]}

    for held in ("val", "test"):
        assert not kitchens(held) & kitchens("dev"), held
        assert not categories(held) & categories("dev"), held
    assert not categories("val") & categories("test")


@pytest.mark.parametrize("part", ["dev", "val", "test"])
def test_every_pick_setup_pins_its_models_and_names_its_target(part):
    for s in pick_parts()[part]:
        cats = [o["category"] for o in s.scene["objects"]]
        assert all("model" in o for o in s.scene["objects"]) and len(set(cats)) == 3, s.name
        assert s.raw_tasks["pick"]["object"] == cats[0] == s.attributes["object"], s.name
        assert json.dumps(s.scene)  # plain data
