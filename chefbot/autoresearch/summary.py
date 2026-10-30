"""Summaries of episode results: overall, and grouped by setup attributes. (agentic-skills-exploration's evals/summary.py.)"""

from __future__ import annotations

from chefbot.autoresearch import setups

RATES = ("success",)
MEANS = ("lift_cm", "max_lift_cm", "robot_time_s", "sim_s", "think_s", "model_s", "model_usd", "wall_s")


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    n = len(results)

    def mean(k):
        v = [r[k] for r in ok if r.get(k) is not None]
        return round(sum(v) / len(v), 4) if v else None

    out = {"episodes": n, "harness_errors": n - len(ok)}
    for k in RATES:
        scored = [r for r in results if r.get(k) is not None]
        out[k] = round(sum(bool(r[k]) for r in scored) / len(scored), 3) if scored else None
    out |= {f"mean_{k}": mean(k) for k in MEANS if any(k in r for r in ok)}
    out["sandbox_blocked_total"] = sum(r.get("sandbox_blocked", 0) for r in ok)
    return out


def grouped(results: list[dict], group_by: list[str]) -> dict:
    """{attribute: {value: summary}} over the setups' attributes."""
    attrs = {}
    out = {}
    for key in group_by:
        groups = {}
        for r in results:
            s = r.get("setup")
            if s not in attrs:
                attrs[s] = setups.load(s).attributes if s else {}
            groups.setdefault(str(attrs[s].get(key, "?")), []).append(r)
        out[key] = {v: summarize(rs) for v, rs in sorted(groups.items())}
    return out


def table(summary: dict, groups: dict) -> str:
    """A markdown table: overall, then each group."""
    cols = [k for k in ("success", "mean_lift_cm", "mean_robot_time_s", "mean_model_usd") if summary.get(k) is not None]
    head = "| | " + " | ".join(cols) + " | episodes |\n|---|" + "---|" * (len(cols) + 1) + "\n"
    row = lambda name, s: f"| {name} | " + " | ".join(str(s.get(c)) for c in cols) + f" | {s['episodes']} |\n"  # noqa: E731
    out = head + row("all", summary)
    for key, g in groups.items():
        for v, s in g.items():
            out += row(f"{key}={v}", s)
    return out
