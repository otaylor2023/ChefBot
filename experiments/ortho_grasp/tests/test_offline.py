"""Offline tests (spec §17.2): no model, no simulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from experiments.ortho_grasp.evaluate import e1_geometric
from experiments.ortho_grasp.observe import CupGeometry
from experiments.ortho_grasp.render.plots import specs_for
from experiments.ortho_grasp.resolve import lift, resolve_p1, resolve_p2

CENTER = np.array([0.70, 0.0])
R_OUT, WALL, BOTTOM, HEIGHT, COUNTER = 0.04, 0.004, 0.92, 0.10, 0.92


def synthetic_cup(pieces: int = 24) -> CupGeometry:
    """A hollow cylinder (r 40 mm, wall 4 mm, 100 mm tall) on the counter, as
    convex wall wedges plus a floor disc, like a decomposed collision mesh."""
    hulls, surface = [], []
    rng = np.random.default_rng(0)
    for k in range(pieces):
        t0, t1 = 2 * np.pi * k / pieces, 2 * np.pi * (k + 1) / pieces
        corners = []
        for t in (t0, t1):
            for r in (R_OUT - WALL, R_OUT):
                for z in (BOTTOM, BOTTOM + HEIGHT):
                    corners.append([CENTER[0] + r * np.cos(t), CENTER[1] + r * np.sin(t), z])
        corners = np.array(corners)
        hulls.append(ConvexHull(corners).equations)
        t = rng.uniform(t0, t1, 400)
        z = rng.uniform(BOTTOM, BOTTOM + HEIGHT, 400)
        surface.append(np.stack([CENTER[0] + R_OUT * np.cos(t), CENTER[1] + R_OUT * np.sin(t), z], axis=1))
        surface.append(np.stack([CENTER[0] + (R_OUT - WALL) * np.cos(t), CENTER[1] + (R_OUT - WALL) * np.sin(t), z], axis=1))
    t = np.linspace(0, 2 * np.pi, 64)
    disc = np.array([[CENTER[0] + r * np.cos(a), CENTER[1] + r * np.sin(a), z] for a in t for r in (0.0, R_OUT - WALL) for z in (BOTTOM, BOTTOM + 0.005)])
    hulls.append(ConvexHull(disc).equations)
    full = np.concatenate(surface)
    return CupGeometry(full, full[:100], hulls, COUNTER, BOTTOM + HEIGHT, BOTTOM)


# -- plot spec ----------------------------------------------------------------------


def test_plot_spec_round_trip():
    cup = synthetic_cup()
    for spec in specs_for(cup.box, 0.20).values():
        hv = np.random.default_rng(1).uniform([spec.h_range[0], spec.v_range[0]], [spec.h_range[1], spec.v_range[1]], (200, 2))
        back = spec.to_world(spec.to_pixel(hv))
        assert np.abs(back - hv).max() < 0.0005
    specs = specs_for(cup.box, 0.20)
    assert specs["top"].m_per_px == pytest.approx(specs["side"].m_per_px)  # same scale in both views


# -- lifting ------------------------------------------------------------------------------


def test_lift_takes_x_y_from_top_and_z_from_side():
    pts, dis = lift({"A": (0.70, 0.05)}, {"A": (0.705, 1.00)}, ["A"])
    assert pts["A"] == pytest.approx([0.70, 0.05, 1.00])
    assert dis["A"] == pytest.approx(0.005)
    pts, dis = lift({"A": (0.70, 0.05)}, {"A": 0.98}, ["A"])  # sequential: z only
    assert pts["A"] == pytest.approx([0.70, 0.05, 0.98]) and dis == {}


# -- P1 resolution ------------------------------------------------------------------------


def marks(A, B, F, side_dx=0.0):
    top = {k: (v[0], v[1]) for k, v in zip("ABF", (A, B, F))}
    side = {k: (v[0] + side_dx, v[2]) for k, v in zip("ABF", (A, B, F))}
    return lift(top, side, ["A", "B", "F"])


RIM = BOTTOM + HEIGHT - 0.02
GOOD = ([0.70 - R_OUT - 0.008, 0.0, RIM], [0.70 - R_OUT + WALL + 0.008, 0.0, RIM], [0.70 - R_OUT + WALL / 2, 0.0, RIM + 0.156])


def test_p1_consistent_marks_resolve():
    r = resolve_p1(*marks(*GOOD))
    assert r.ok and not r.flags["gs_large"] and not r.flags["x_disagree"]
    assert r.pose[:3, 2] == pytest.approx([0, 0, -1], abs=1e-6)  # approach straight down
    assert r.width == pytest.approx(WALL + 0.016)


def test_p1_flange_not_above_fingers_is_gs_large():
    A, B, F = GOOD
    F = [F[0] + 0.06, F[1], F[2]]  # flange off to the side: approach tilted toward the closing axis
    r = resolve_p1(*marks(A, B, F))
    assert r.flags["gs_large"] and r.metrics["gs_angle_deg"] > 10


def test_p1_cross_view_disagreement_is_flagged():
    r = resolve_p1(*marks(*GOOD, side_dx=0.02))
    assert r.ok and r.flags["x_disagree"]


def test_p1_parallel_is_rejected():
    A, B, _ = GOOD
    F = [B[0] + 0.15, B[1], B[2]]  # flange along the closing line
    r = resolve_p1(*marks(A, B, F))
    assert not r.ok and r.reason == "parallel"


def test_p1_from_below_is_rejected():
    A, B, F = GOOD
    F = [F[0], F[1], RIM - 0.156]
    r = resolve_p1(*marks(A, B, F))
    assert not r.ok and r.reason == "from_below"


def test_p2_picks_straight_down_when_clear():
    cup = synthetic_cup()
    A, B, _ = GOOD
    pts, dis = lift({"A": A[:2], "B": B[:2]}, {"A": (A[0], A[2]), "B": (B[0], B[2])}, ["A", "B"])
    r = resolve_p2(pts, dis, cup.full, COUNTER)
    assert r.ok and r.pose[:3, 2] == pytest.approx([0, 0, -1], abs=1e-6)


# -- E1 ------------------------------------------------------------------------------------


def test_e1_rim_grasp_passes():
    cup = synthetic_cup()
    A, B, _ = GOOD
    out = e1_geometric.evaluate(np.array(A), np.array(B), WALL + 0.016 + 0.004, cup)
    assert out["straddle"] and out["clear_tips"] and out["width_ok"] and out["height_ok"] and out["e1_valid"]


def test_e1_both_fingers_outside_the_wall_fails_straddle():
    cup = synthetic_cup()
    A = np.array([0.70 - R_OUT - 0.03, 0.0, RIM])
    B = np.array([0.70 - R_OUT - 0.01, 0.0, RIM])
    out = e1_geometric.evaluate(A, B, 0.03, cup)
    assert not out["straddle"] and not out["e1_valid"]


def test_e1_fingertip_inside_the_wall_fails_clear_tips():
    cup = synthetic_cup()
    A = np.array([0.70 - R_OUT - 0.008, 0.0, RIM])
    B = np.array([0.70 - R_OUT + WALL / 2, 0.0, RIM])  # in the wall
    out = e1_geometric.evaluate(A, B, 0.02, cup)
    assert not out["clear_tips"] and not out["e1_valid"]


def test_e1_body_grasp_across_the_cup_straddles():
    cup = synthetic_cup()
    A = np.array([0.70 - R_OUT - 0.006, 0.0, RIM - 0.03])
    B = np.array([0.70 + R_OUT + 0.006, 0.0, RIM - 0.03])
    out = e1_geometric.evaluate(A, B, 0.085, cup)
    assert out["straddle"] and out["runs"] == 2 and out["thickness"] == pytest.approx(2 * R_OUT, abs=0.002)


def test_flange_correction_keeps_the_fingers_and_turns_the_approach():
    from experiments.ortho_grasp.resolve import correct_flange

    cup = synthetic_cup()
    A, B, F = (np.array(p) for p in GOOD)
    F_bad = F + np.array([0.0, 0.10, -0.05])  # flange off to the side and low: the body would sweep the counter side
    r = correct_flange(A, B, F_bad, cup.full, COUNTER, reachable=lambda rr: True)
    assert r.ok
    a, b = r.fingertips
    assert a == pytest.approx(A) and b == pytest.approx(B)  # fingers untouched
    assert r.pose[:3, 0] == pytest.approx((B - A) / np.linalg.norm(B - A))
    assert r.metrics["flange_correction_deg"] > 10


def test_flange_correction_gives_up_when_nothing_is_reachable():
    from experiments.ortho_grasp.resolve import correct_flange

    cup = synthetic_cup()
    r = correct_flange(*(np.array(p) for p in GOOD), cup.full, COUNTER, reachable=lambda rr: False)
    assert not r.ok and r.reason == "no_valid_flange"
