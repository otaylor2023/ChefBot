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


def straight(heading: float, move_deg: float, up: str, n: int = 60) -> list[dict]:
    d = heading + math.radians(move_deg)
    return [motive_sample(i * 0.01, 1.0 + 0.01 * i * math.cos(d), 2.0 + 0.01 * i * math.sin(d), heading, up) for i in range(n)]


def drives(tmp_path, up: str, moves=("forward",)):
    heading = 0.7
    rows_by = {m: straight(heading, B.MOVES_DEG[m], up, 100 if m == "forward" else 60) for m in moves}
    rows_by["spin"] = [motive_sample(i * 0.01, 1.5, -0.5, 2 * math.pi * i / 200, up) for i in range(200)]
    paths = {}
    for name, rows in rows_by.items():
        p = tmp_path / f"{name}_{up}.jsonl"
        p.write_text("\n".join([json.dumps({"header": {}})] + [json.dumps(r) for r in rows]))
        paths[name] = p
    return paths


@pytest.mark.parametrize("up", ["y", "z"])
def test_solve_finds_the_up_axis_offset_and_marker_place(tmp_path, up):
    p = drives(tmp_path, up)
    out = B.solve(p["forward"], p["spin"], [])
    assert out["up_axis"] == up and out["up_axis_agree"]
    assert out["up_from_rotation"]["off_deg"] < 1.0 and out["up_from_positions"]["off_deg"] < 1.0
    assert out["forward"]["offset_deg"] == pytest.approx(30.0, abs=0.5)
    assert out["forward"]["distance_m"] == pytest.approx(0.99, abs=0.01)
    np.testing.assert_allclose(out["spin"]["marker_xy"], MARKER_XY, atol=0.002)
    assert out["spin"]["turned_deg"] > 350 and out["spin"]["turned_left"]


@pytest.mark.parametrize("up", ["y", "z"])
def test_all_four_moves_agree_and_the_frame_is_not_mirrored(tmp_path, up):
    p = drives(tmp_path, up, moves=("forward", "back", "left", "right"))
    out = B.solve(p["forward"], p["spin"], [], p["back"], p["left"], p["right"])
    assert out["marker_yaw_offset_deg"] == pytest.approx(30.0, abs=0.5)
    assert max(abs(d) for d in out["offset_disagreement_deg"].values()) < 1.0
    assert out["mirrored"] is False


def test_a_mirrored_frame_shows_up_as_left_and_right_180_off(tmp_path):
    # The wrong sign of up mirrors the floor: forward and back still agree, left and right come out 180 deg off.
    p = drives(tmp_path, "y", moves=("forward", "back", "left", "right"))
    rows = {m: B.live(B.load(p[m])) for m in ("forward", "back", "left", "right")}
    X, Y, _ = B.floor_frame(1)
    est = {m: B.heading_offset(rows[m], X, -Y, B.MOVES_DEG[m])["offset_deg"] for m in rows}  # Y flipped: mirrored
    assert abs(B.wrap(math.radians(est["forward"] - est["back"]))) < math.radians(1)
    assert abs(abs(math.degrees(B.wrap(math.radians(est["left"] - est["forward"])))) - 180) < 1


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
    p = drives(tmp_path, "z")
    out = B.solve(p["forward"], p["spin"], [])
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


def test_a_sloppy_stick_reaches_the_base_as_one_clean_motion():
    from chefbot.onboard.bringup.gamepad import one_axis

    assert one_axis(-0.9, 0.2, 0.3) == (-0.9, 0.0, 0.0)  # mostly up (forward): sideways and turn dropped
    assert one_axis(0.15, -0.7, 0.0) == (0.0, -0.7, 0.0)  # mostly sideways
    assert one_axis(0.02, 0.03, 0.6) == (0.0, 0.0, 0.6)  # left stick centred: the turn alone
    assert one_axis(0.3, 0.0, 0.8) == (0.3, 0.0, 0.0)  # never both


def test_one_axis_joystick_passes_buttons_through():
    from chefbot.onboard.bringup.gamepad import OneAxisJoystick

    class Pad:
        axes = {0: 0.2, 1: -0.9, 3: 0.4}

        def get_axis(self, i):
            return self.axes.get(i, 0.0)

        def get_button(self, i):
            return i == 4

    pad = OneAxisJoystick(Pad())
    assert (pad.get_axis(1), pad.get_axis(0), pad.get_axis(3)) == (-0.9, 0.0, 0.0)
    assert pad.get_button(4) and not pad.get_button(5)


def test_a_curved_or_turning_move_is_reported_and_not_used(tmp_path):
    p = drives(tmp_path, "y", moves=("forward", "back", "left", "right"))
    # Re-record "left" with the base turning 15 deg on the way (a sloppy stick without --one-axis).
    rows = [motive_sample(i * 0.01, 1.0, 2.0 + 0.01 * i, 0.7 + math.radians(15) * i / 60, "y") for i in range(60)]
    p["left"].write_text("\n".join(json.dumps(r) for r in rows))
    out = B.solve(p["forward"], p["spin"], [], p["back"], p["left"], p["right"])
    assert out["left"]["ok"] is False and out["moves_used"] == ["back", "forward", "right"]
    assert out["marker_yaw_offset_deg"] == pytest.approx(30.0, abs=0.5)


def test_the_line_fit_ignores_a_wobble_at_the_end(tmp_path):
    p = drives(tmp_path, "y")
    rows = B.live(B.load(p["forward"]))
    rows[-1] = dict(rows[-1], pos=(np.array(rows[-1]["pos"]) + [0.0, 0.0, 0.03]).tolist())  # 3 cm sideways at the stop
    X, Y, _ = B.floor_frame(1)
    assert B.heading_offset(rows, X, Y)["offset_deg"] == pytest.approx(30.0, abs=0.6)


def test_echo_shows_what_gamepad_teleop_would_send():
    from chefbot.onboard.bringup.gamepad import command

    # Stick pushed up and a bit right (pygame: up is negative axis 1, right is positive axis 0), right stick nudged.
    vx, vy, w = command(-1.0, 0.3, 0.4, vel=0.2, yaw_vel=0.5, snap=True)
    assert vx == pytest.approx(0.2) and vy == 0.0 and w == 0.0  # exactly forward at the limit
    vx, vy, w = command(-1.0, 0.3, 0.4, vel=0.2, yaw_vel=0.5, snap=False)
    assert vy < 0 and w < 0  # without snapping: right and turning right too
    assert command(0.0, -1.0, 0.0, 0.2, 0.5, True) == pytest.approx((0.0, 0.2, 0.0))  # stick left: left
    assert command(0.0, 0.0, -1.0, 0.2, 0.5, True) == pytest.approx((0.0, 0.0, 0.5))  # right stick left: turn left


def test_the_wheels_steering_at_the_start_of_a_move_does_not_bias_it(tmp_path):
    # A swerve base lurches sideways while its modules steer to the new direction: 4 cm over the first 10 cm here.
    heading, d = 0.7, 0.7  # driving forward
    side = d + math.pi / 2
    rows = []
    for i in range(100):
        a = 0.01 * i
        lurch = 0.04 * min(a / 0.1, 1.0)
        x = 1.0 + a * math.cos(d) + lurch * math.cos(side)
        y = 2.0 + a * math.sin(d) + lurch * math.sin(side)
        rows.append(motive_sample(0.01 * i, x, y, heading, "y"))
    X, Y, _ = B.floor_frame(1)
    out = B.heading_offset(rows, X, Y)
    assert out["offset_deg"] == pytest.approx(30.0, abs=0.5)  # end-to-end it would be ~2.3 deg off
    assert out["ok"] and out["steady_samples"] < 100
