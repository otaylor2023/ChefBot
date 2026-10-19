"""Score an explored map against the true semantic map.

    python -m chefbot.explore.score runs/<run>/explored_map.yaml maps/robocasa_layout001_style001/semantic_map.yaml

Fixtures must be the same. Each true object must be found: same name, same
`in`, and within `location_tolerance` (xy) of its true location. Objects the
truth does not have are false finds. 100% = every true object found and no
false finds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from chefbot.agent.world.map import label_of

CONFIG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())


def _same_name(a: str, b: str) -> bool:
    """Names match with underscores read as spaces (bottled_water = bottled water)."""
    return a.replace("_", " ") == b.replace("_", " ")


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def score(explored: dict, truth: dict, tolerance: float | None = None) -> dict:
    tol = CONFIG["location_tolerance"] if tolerance is None else tolerance
    got_f, true_f = explored.get("fixtures") or {}, truth.get("fixtures") or {}
    fixtures_ok = list(got_f) == list(true_f) and all(
        got_f[n].get("location") == true_f[n].get("location") and got_f[n].get("facing") == true_f[n].get("facing")
        for n in true_f
    )
    got, want = dict(explored.get("objects") or {}), truth.get("objects") or {}
    # Pair each true object with the nearest unpaired explored object of its label.
    pairs = sorted(
        (_xy_dist(g["location"], w["location"]), wn, gn)
        for wn, w in want.items() for gn, g in got.items() if label_of(gn) == label_of(wn)
    )  # fmt: skip
    match: dict[str, str] = {}
    for _, wn, gn in pairs:
        if wn not in match and gn not in match.values():
            match[wn] = gn
    rows = []
    for wn, w in want.items():
        gn = match.get(wn)
        if gn is None:
            rows.append({"object": wn, "status": "missing", "true_in": w.get("in")})
            continue
        g = got[gn]
        d = _xy_dist(g["location"], w["location"])
        problems = []
        if not _same_name(gn, wn):
            problems.append(f"named {gn}")
        if g.get("in") != w.get("in"):
            problems.append(f"in {g.get('in')}, truly in {w.get('in')}")
        if d > tol:
            problems.append(f"{d:.2f} m off")
        rows.append({"object": wn, "status": "ok" if not problems else "wrong", "problems": problems,
                     "found_as": gn, "error_m": round(d, 3)})  # fmt: skip
    false_finds = sorted(set(got) - set(match.values()))
    found = sum(r["status"] == "ok" for r in rows)
    total = len(want)
    return {
        "score": found / total if total else 1.0,
        "found": found,
        "total": total,
        "false_finds": false_finds,
        "fixtures_match": fixtures_ok,
        "perfect": fixtures_ok and found == total and not false_finds,
        "objects": rows,
        "tolerance_m": tol,
    }


def describe(result: dict) -> str:
    lines = [f"explored map: {result['found']}/{result['total']} objects right"
             + (f", {len(result['false_finds'])} false finds" if result["false_finds"] else "")
             + ("" if result["fixtures_match"] else ", FIXTURES DIFFER")
             + (" -> 100% match" if result["perfect"] else "")]  # fmt: skip
    for r in result["objects"]:
        if r["status"] == "ok":
            lines.append(f"  ok       {r['object']} ({r['error_m']:.2f} m)")
        elif r["status"] == "missing":
            lines.append(f"  MISSING  {r['object']} (in {r['true_in']})")
        else:
            lines.append(f"  WRONG    {r['object']}: {'; '.join(r['problems'])}")
    for name in result["false_finds"]:
        lines.append(f"  FALSE    {name} (not in the true map)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("explored", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = score(yaml.safe_load(args.explored.read_text()), yaml.safe_load(args.truth.read_text()), args.tolerance)
    print(json.dumps(result, indent=2) if args.json else describe(result))


if __name__ == "__main__":
    main()
