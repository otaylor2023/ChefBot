"""Plan store: status rules and error codes, the edit log, the reference plan (AGENT_SPEC §4)."""

import pytest

from chefbot.agent.plan.model import Plan
from chefbot.agent.plan.store import PlanError, PlanStore

PLAN = {
    "goal": "Fried egg",
    "steps": [
        {"id": "s1", "goal": "Get the egg", "done_when": "the egg is held"},
        {"id": "s2", "goal": "Pan on a burner", "done_when": "the pan is on a burner"},
        {"id": "s3", "goal": "Cook the egg", "done_when": "the egg is set", "depends_on": ["s1", "s2"]},
    ],
}


def store(now=lambda: 0.0):
    return PlanStore(Plan.from_dict(PLAN), now)


def code(fn, *args, **kwargs) -> str:
    with pytest.raises(PlanError) as err:
        fn(*args, **kwargs)
    return err.value.code


def test_one_step_active_at_a_time():
    s = store()
    s.set_status("s1", "active")
    assert code(s.set_status, "s2", "active") == "another_step_active"
    s.set_status("s1", "pausing")
    assert code(s.set_status, "s2", "active") == "another_step_active"  # pausing counts
    s.set_status("s1", "paused", "egg still in the fridge")
    s.set_status("s2", "active")
    assert s.active().id == "s2"


def test_dependencies_must_be_done_or_skipped():
    s = store()
    assert code(s.set_status, "s3", "active") == "dependencies_not_met"
    s.complete("s1")
    s.skip("s2", "pan already on the burner")
    s.set_status("s3", "active")


def test_done_only_through_complete_and_reasons_are_required():
    s = store()
    assert code(s.set_status, "s1", "done") == "use_complete"
    s.set_status("s1", "active")
    assert code(s.set_status, "s1", "paused") == "note_required"
    assert code(s.set_status, "s1", "failed") == "note_required"
    assert code(s.skip, "s2", "") == "note_required"
    assert code(s.set_status, "s1", "cooked") == "invalid_status"
    assert code(s.set_status, "s9", "active") == "unknown_step"


def test_transitions_allowed_by_the_spec():
    s = store()
    s.set_status("s1", "active")
    s.set_status("s1", "paused", "resume: open the fridge again")  # active -> paused directly
    s.set_status("s1", "active")
    s.set_status("s1", "waiting")
    s.set_status("s1", "active")
    s.complete("s1")
    assert code(s.set_status, "s1", "active") == "invalid_transition"  # done is final
    assert code(s.complete, "s1") == "invalid_transition"


def test_complete_uses_the_last_note_as_the_log_line():
    s = store()
    s.set_status("s1", "active")
    s.add_note("s1", "egg picked from the fridge shelf")
    assert s.complete("s1").log == "egg picked from the fridge shelf"
    s.set_status("s2", "active")
    assert s.complete("s2").log == "completed"


def test_edit_log_records_every_change_and_the_reference_never_changes():
    t = [0.0]
    s = store(lambda: t[0])
    t[0] = 1.0
    s.set_status("s1", "active")
    t[0] = 2.0
    added = s.add_step("s1", "Move the pot off burner 1", "burner_1 is free", "a pot is on burner_1")
    s.skip("s2", "using burner 2")
    assert added.id == "s1a" and [x.id for x in s.live.steps] == ["s1", "s1a", "s2", "s3"]
    assert added.added_because == "a pot is on burner_1"
    assert [(e.t, e.step, e.change) for e in s.edits] == [
        (1.0, "s1", "status: pending -> active"),
        (2.0, "s1a", "added after s1"),
        (2.0, "s2", "status: pending -> skipped"),
    ]
    assert [x.id for x in s.reference.steps] == ["s1", "s2", "s3"]
    assert all(x.status == "pending" for x in s.reference.steps)


def test_plan_files_are_validated():
    with pytest.raises(ValueError, match="unknown steps"):
        Plan.from_dict({"goal": "x", "steps": [{"id": "a", "goal": "g", "done_when": "d", "depends_on": ["z"]}]})
    with pytest.raises(ValueError, match="duplicate"):
        Plan.from_dict({"goal": "x", "steps": [{"id": "a", "goal": "g", "done_when": "d"}] * 2})
