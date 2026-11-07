from pathlib import Path

import pytest

from chefbot.scripts.real_nav import counter_box, sweep_targets

MAP = Path(__file__).resolve().parents[3] / "maps" / "src_kitchen"


def test_kitchen_sweep_targets_sit_on_their_sides_at_the_trained_inset():
    targets, boxes = sweep_targets(MAP, MAP / "sweep.yaml")
    assert len(targets) == len(boxes) >= 5
    from chefbot.robot.nav.goal import counter_park_candidates

    for t, b in zip(targets, boxes):
        cands = counter_park_candidates(*b[:3], t)
        assert cands, f"no stance by the pick rule for {t} on {b[3]}"
        assert any(0.10 <= info["inset_m"] <= 0.16 for _, info in cands)


def test_sweep_refuses_an_x_off_the_side(tmp_path):
    f = tmp_path / "s.yaml"
    f.write_text("stops:\n  - {counter: kitchen_island_countertop, side: south, x: 9.0}\n")
    with pytest.raises(SystemExit):
        sweep_targets(MAP, f)
    assert counter_box(MAP, "kitchen_island_countertop")
