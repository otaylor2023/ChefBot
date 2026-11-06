"""RecipeBench: perturbations are deterministic and the per-instance tests catch the failures they are for."""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from bench.checks import check, quote_f1  # noqa: E402
from bench.perturb import INJECTION, PERTURBATIONS, drop_cooking, inject_instruction  # noqa: E402
from bench.build import cooking_quotes, head_word  # noqa: E402
from recipe_ir.adapters import from_text  # noqa: E402
from recipe_ir.recipe_pipeline import validate  # noqa: E402
from recipe_ir.schema import Recipe  # noqa: E402


def gold(name):
    text = (HERE / "recipes" / f"{name}.txt").read_text()
    return text, validate(Recipe.model_validate_json((HERE / "recipes" / "gold" / f"{name}.gold.json").read_text()), from_text(text))


def test_perturbations_are_deterministic_and_change_text():
    text, _ = gold("scrambled_eggs")
    for name, fn in PERTURBATIONS.items():
        a, b = fn(text, seed=0), fn(text, seed=0)
        assert a == b, name
        if a is not None:
            assert a[0] != text, name


def test_heat_level_still_stated_elsewhere_is_not_applicable():
    text, _ = gold("scrambled_eggs")          # "Low heat and patience..." stays in the intro
    assert PERTURBATIONS["no_heat_level"](text) is None


def test_units_convert_celsius_to_fahrenheit():
    text, _ = gold("blueberry_muffins")
    new, exp = PERTURBATIONS["units"](text)
    assert "374°F" in new and exp["expected_c"] == [190]


def test_gold_resolves_its_own_base_instance():
    text, v = gold("scrambled_eggs")
    inst = {"expect": {"kind": "base", "gold": True, "status": "valid"}}
    assert check(inst, v, v)["resolved"]


def test_onion_egg_valid_status_is_wrong():
    text, v = gold("onion_egg")            # human answer: recipe_findings (egg never cooked)
    inst = {"expect": {"kind": "base", "gold": True, "status": "valid"}}
    out = check(inst, v, v)
    assert not out["resolved"] and "status" in out["reasons"][0]


def test_planted_gap_removes_the_cooking_sentence():
    text, v = gold("scrambled_eggs")
    cq = cooking_quotes(v)
    assert "egg" in cq
    new, exp = drop_cooking(text, {"cooking_quotes": cq, "words": {i.id: head_word(i.name) for i in v.recipe.ingredients}})
    assert "stir slowly with a spatula" not in new and exp["word"] == "egg"


def test_injection_check_flags_a_followed_instruction():
    text, v = gold("scrambled_eggs")
    new, exp = inject_instruction(text)
    r = v.recipe.model_copy(deep=True)
    r.ingredients[0].name = "bleach"            # pretend the model obeyed the page
    vv = validate(r, from_text(text))
    out = check({"expect": exp}, vv, v)
    assert not out["resolved"] and any("injected" in x for x in out["reasons"])
    assert INJECTION in new


def test_quote_f1_identical_is_one():
    _, v = gold("blueberry_muffins")
    assert quote_f1(v.recipe, v.recipe) == 1.0


def test_head_word():
    assert head_word("large eggs") == "egg" and head_word("boneless chicken thighs") == "chicken"
    assert head_word("dry-packed sea scallops") == "scallop"


def test_doneness_temperature_counts_for_units():
    _, v = gold("scrambled_eggs")
    r = v.recipe.model_copy(deep=True)
    from recipe_ir.schema import Until
    r.steps[5].until = Until(kind="temp", target="49°C", raw="center registers 49°C")
    vv = validate(r, from_text((HERE / "recipes" / "scrambled_eggs.txt").read_text()))
    out = check({"expect": {"kind": "temperature", "expected_c": [49], "tolerance_c": 3}}, vv, v)
    assert out["metrics"]["temps_c"] == [49.0]
