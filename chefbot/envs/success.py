"""Task success from ground truth: a task's rules against the observer's relation lines.
Shared by every env (symbolic, RoboCasa).

    require: [regex, ...]        each must match some line
    forbid: [regex, ...]         none may match
    expect_tools / forbid_tools  tools that must (not) have been called successfully
    at_most: {regex: n}          at most n lines match
    any_of: [rules, ...]         alternatives: the first that holds wins
"""

from __future__ import annotations

import re


def check_success(rules: dict, lines: list[str], tools_called: list[str] = ()) -> dict:
    if "any_of" in rules:
        results = [_check(alt, lines, tools_called) for alt in rules["any_of"]]
        return next((r for r in results if r["success"]), results[0])
    return _check(rules, lines, tools_called)


def _check(rules: dict, lines: list[str], tools_called) -> dict:
    missing = [p for p in rules.get("require", []) if not any(re.search(p, line) for line in lines)]
    missing += [f"call {t}" for t in rules.get("expect_tools", []) if t not in tools_called]
    broken = [p for p in rules.get("forbid", []) if any(re.search(p, line) for line in lines)]
    broken += [f"call {t}" for t in rules.get("forbid_tools", []) if t in tools_called]
    for pattern, most in (rules.get("at_most") or {}).items():
        if sum(bool(re.search(pattern, line)) for line in lines) > int(most):
            broken.append(f"more than {most} of {pattern}")
    return {"success": not missing and not broken, "missing": missing, "forbidden_present": broken}
