"""Position accuracy (explore/accuracy.py): matching and metrics on small hand-made maps, and
regression bounds on copies of real runs (fixtures/accuracy/)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chefbot.explore.accuracy import Run, load_run, match, object_errors, summarize

FIXTURES = Path(__file__).parent / "fixtures" / "accuracy"


def obj(x, y, inside="counter_1"):
    return {"location": [x, y], "in": inside}


def test_match_pairs_by_label_nearest_first_within_the_radius():
    truth = {"mug_1": obj(0.0, 0.0), "mug_2": obj(1.0, 0.0), "egg": obj(2.0, 0.0), "pan": obj(3.0, 0.0)}
    explored = {"mug_1": obj(0.95, 0.0), "mug_2": obj(0.1, 0.0), "egg": obj(2.0, 0.6), "bowl": obj(3.0, 0.0)}
    assert match(explored, truth) == {"mug_1": "mug_2", "mug_2": "mug_1", "egg": None, "pan": None}
    assert match(explored, truth, radius=1.0)["egg"] == "egg"


def test_errors_against_origin_and_centre_and_the_fixture():
    truth = {"pan": obj(1.0, 0.0), "egg": obj(0.0, 0.0, "fridge")}
    explored = {"pan": obj(1.1, 0.0), "egg": obj(0.0, 0.03, "counter_1")}
    run = Run("r", "vision", 1, explored, truth,
              explored_3d={"pan": [1.1, 0.0, 1.0], "egg": [0.0, 0.03, 0.94]},
              truth_3d={"pan": {"origin": [1.0, 0.0, 0.98], "center": [1.1, 0.0, 1.0]},
                        "egg": {"origin": [0.0, 0.0, 0.9], "center": [0.0, 0.0, 0.9]}})  # fmt: skip
    rows = {r["object"]: r for r in object_errors(run)}
    pan, egg = rows["pan"], rows["egg"]
    assert pan["xy"] == pytest.approx(0.1) and pan["xy_center"] == pytest.approx(0.0, abs=1e-9)
    assert pan["offset_xy"] == pytest.approx(0.1) and pan["z"] == pytest.approx(0.02)
    assert pan["d3"] == pytest.approx((0.1**2 + 0.02**2) ** 0.5) and pan["in_ok"]
    assert egg["xy"] == pytest.approx(0.03) and egg["z"] == pytest.approx(0.04) and not egg["in_ok"]


def test_summary_statistics_and_bands():
    rows = [{"found_as": "a", "xy": v, "in_ok": i < 3} for i, v in enumerate((0.01, 0.04, 0.08, 0.12, 0.30))]
    rows.append({"found_as": None})
    s = summarize(rows)
    assert (s["objects"], s["matched"], s["n"]) == (6, 5, 5)
    assert s["median"] == pytest.approx(0.08) and s["max"] == pytest.approx(0.30)
    assert s["mean"] == pytest.approx(0.11) and s["p90"] == pytest.approx(0.228)
    assert (s["within_5cm"], s["within_10cm"], s["within_15cm"]) == (0.4, 0.6, 0.8)
    assert s["fixture_ok"] == pytest.approx(0.6)


# -- regression: copies of real runs ---------------------------------------------------------
# Bounds (cm) per detector, from the runs of 2026-10-25 with some margin: (min matched share,
# max xy median, max xy p90, max 3D median, min share within 15 cm). The exact detector is the
# floor: its error is the true map's reference offset (body origin vs the geometry's middle).
BOUNDS = {
    "groundtruth": (0.80, 1.0, 12.0, 2.0, 0.90),
    "vision": (0.45, 6.0, 16.0, 7.0, 0.85),
    "owl": (0.40, 8.0, 22.0, 8.0, 0.75),
}


def _cases():
    return sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) if FIXTURES.exists() else []


def check_bounds(run: Run) -> None:
    rows = object_errors(run)
    s, s3 = summarize(rows), summarize(rows, "d3")
    matched, median, p90, median_3d, within = BOUNDS[run.detector]
    assert s["matched"] / s["objects"] >= matched, s
    assert 100 * s["median"] <= median and 100 * s["p90"] <= p90, s
    assert 100 * s3["median"] <= median_3d, s3
    assert s["within_15cm"] >= within, s


@pytest.mark.parametrize("case", _cases())
def test_real_runs_stay_within_their_bounds(case):
    check_bounds(load_run(FIXTURES / case))


def test_the_exact_detector_error_is_the_reference_offset():
    for case in (c for c in _cases() if c.startswith("groundtruth")):
        for r in object_errors(load_run(FIXTURES / case)):
            if "xy_center" in r:
                assert r["xy_center"] <= 0.01, (case, r)  # it sees the middle of the geometry exactly


@pytest.mark.slow
@pytest.mark.robocasa
@pytest.mark.skipif(not os.environ.get("CHEFBOT_SLOW"), reason="a simulator run: set CHEFBOT_SLOW=1")
def test_owl_on_the_default_kitchen_stays_within_its_bounds(tmp_path):
    pytest.importorskip("transformers")
    from chefbot.scripts.explore import run

    result = run("robocasa", "scripted", "owl", out=tmp_path, quiet=True)
    check_bounds(load_run(result["run_dir"]))
