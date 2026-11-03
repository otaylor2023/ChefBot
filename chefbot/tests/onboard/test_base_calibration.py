"""onboard/calibration/base.py against synthetic drives, streamed Y-up (Motive's default) and Z-up."""

import json
import math

import numpy as np
import pytest

from chefbot.onboard import keys as K
from chefbot.onboard.calibration import base as B
from chefbot.onboard.frames import BaseFrames, yaw_of_quat_xyzw

OFFSET = math.radians(30.0)  # the marker's heading on the base
MARKER_XY = (0.12, -0.05)  # the marker's origin in the base frame
HEIGHT = 0.58


def motive_sample(t, base_x, base_y, heading, up: str) -> dict:
    """What the publisher would stream for the base at (base_x, base_y, heading) on the floor."""
    up_i = B.AXES.index(up)
    X, Y, U = B.floor_frame(up_i)
    c, s = math.cos(heading), math.sin(heading)
    mx, my = MARKER_XY
    px, py = base_x + c * mx - s * my, base_y + s * mx + c * my
    pos = px * X + py * Y + HEIGHT * U
    theta = heading + OFFSET  # the marker turns about Up with the base
    q = np.zeros(4)
    q[up_i] = math.sin(theta / 2)
    q[3] = math.cos(theta / 2)
    rng = np.random.default_rng(int(t * 1000))
    return {"t": t, "pos": (pos + rng.normal(0, 1e-4, 3)).tolist(), "ori": q.tolist(), "valid": True, "fresh": True,
            "age_s": 0.0, "odom": None}  # fmt: skip


def drives(tmp_path, up: str):
    heading = 0.7
    forward = [motive_sample(i * 0.01, 1.0 + 0.01 * i * math.cos(heading), 2.0 + 0.01 * i * math.sin(heading), heading, up)
               for i in range(100)]  # fmt: skip
    spin = [motive_sample(i * 0.01, 1.5, -0.5, 2 * math.pi * i / 200, up) for i in range(200)]
    paths = []
    for name, rows in (("forward", forward), ("spin", spin)):
        p = tmp_path / f"{name}_{up}.jsonl"
        p.write_text("\n".join([json.dumps({"header": {}})] + [json.dumps(r) for r in rows]))
        paths.append(p)
    return paths


@pytest.mark.parametrize("up", ["y", "z"])
def test_solve_finds_the_up_axis_offset_and_marker_place(tmp_path, up):
    forward, spin = drives(tmp_path, up)
    out = B.solve(forward, spin, [])
    assert out["up_axis"] == up and out["up_axis_agree"]
    assert out["up_from_rotation"]["off_deg"] < 1.0 and out["up_from_positions"]["off_deg"] < 1.0
    assert out["forward"]["offset_deg"] == pytest.approx(30.0, abs=0.5)
    assert out["forward"]["distance_m"] == pytest.approx(0.99, abs=0.01)
    np.testing.assert_allclose(out["spin"]["marker_xy"], MARKER_XY, atol=0.002)
    assert out["spin"]["turned_deg"] > 350


def test_the_z_up_yaw_of_y_up_data_is_only_ever_0_or_180():
    # frames.py's yaw assumes z up. Fed Motive's Y-up stream, a base turning through every heading reads 0 or 180.
    yaws = {round(abs(math.degrees(yaw_of_quat_xyzw(motive_sample(0, 0, 0, h, "y")["ori"])))) for h in np.linspace(-3, 3, 61)}
    assert yaws == {0, 180}


def test_floor_frames_are_right_handed():
    for up in range(3):
        X, Y, U = B.floor_frame(up)
        np.testing.assert_allclose(np.cross(X, Y), U)
    X, Y, _ = B.floor_frame(1)
    np.testing.assert_allclose(X, [1, 0, 0]) and np.testing.assert_allclose(Y, [0, 0, -1])


def test_the_solved_offsets_put_frames_py_on_the_base_for_z_up_data(tmp_path):
    forward, spin = drives(tmp_path, "z")
    out = B.solve(forward, spin, [])
    frames = BaseFrames(math.radians(out["forward"]["offset_deg"]), out["spin"]["marker_xy"])
    s = motive_sample(0, 1.5, -0.5, 1.0, "z")
    x, y, yaw = frames.base_from_mocap(s["pos"], s["ori"])
    assert (x, y, yaw) == pytest.approx((1.5, -0.5, 1.0), abs=0.003)


class FakeRedis:
    def __init__(self):
        self.n = 0

    def mget(self, keys):
        assert keys == (K.MOCAP_POSITION, K.MOCAP_ORIENTATION, K.MOCAP_VALID, K.BASE_POSE)
        self.n += 1
        return [f"[0.1,0.2,{0.3 + 1e-6 * self.n}]", "[0,0,0,1]", "1", None]


def test_record_only_reads_and_logs_raw_samples(tmp_path):
    out = tmp_path / "rec.jsonl"
    B.record(out, hz=200, seconds=0.1, redis_client=FakeRedis())  # FakeRedis has no set: a write would fail
    rows = B.load(out)
    assert len(rows) > 5 and rows[0]["pos"][:2] == [0.1, 0.2] and rows[0]["ori"] == [0, 0, 0, 1]
    assert not rows[0]["fresh"] and rows[-1]["fresh"]  # fresh once the value changes
    assert len(B.live(rows)) == len(rows) - 1
