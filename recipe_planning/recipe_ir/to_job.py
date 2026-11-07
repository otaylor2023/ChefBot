"""actions.json -> a ChefBot job (chefbot/jobs/*.yaml: goal + steps with goal, done_when, depends_on, hint).

The recipe agent stays robot-independent; this is the one place that speaks ChefBot's task format. Only main-path
steps are compiled. Each step keeps the recipe's own words as its goal, gets a checkable `done_when` from the
step's completion cue (or its effect), and its feasibility from the skills file goes into the hint, so the ChefBot
agent knows which steps it can't do before it tries. A job needs a kitchen-specific `symbolic` block (objects and
success) to run in simulation; that is written per kitchen, not here.
"""
from __future__ import annotations
import re
from typing import Any


PAST = {"mix": "mixed", "cook": "cooked", "bake": "baked", "melt": "melted", "beat": "beaten", "season": "seasoned",
        "cut": "cut", "wait": "rested", "cool": "cooled", "transfer": "added"}


def _names(st: dict, names: dict[str, str]) -> str:
    """Ingredient names the step acts on ('eggs and salt'), from what each input is made of."""
    xs: list[str] = []
    for a in st["acts_on"]:
        for x in (a.get("made_of") or [a["ref"]]):
            n = names.get(x, x.replace("_", " "))
            if n not in xs:
                xs.append(n)
    if not xs:
        return "it"
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _where(st: dict) -> str | None:
    dest = (st.get("goes_to") or {}).get("type")
    eq = st["equipment"][0]["type"] if st["equipment"] else None
    return dest or eq


def _goal(st: dict, names: dict[str, str]) -> str:
    """The recipe's words, except for equipment steps whose quote is only a fragment ('over low heat')."""
    v, where = st["verb"], _where(st) or "pan"
    if v == "heat":
        lvl = st["settings"].get("heat")
        return f"Put the {where} on a burner and turn it on" + (f" ({lvl} heat)" if lvl else "")
    if v == "preheat":
        return f"Preheat the oven to {st['settings'].get('temp_c', '?')} C"
    if v == "turn_off":
        return "Turn off the burner"
    q = st["instruction"]
    return q[0].upper() + q[1:]


def _be(what: str) -> str:
    return "are" if " and " in what or (what.endswith("s") and not what.endswith("ss")) else "is"


def _effect(st: dict, names: dict[str, str]) -> str:
    """What is true when the step is done, in the plain words ChefBot's StateChecker judges."""
    v, what, where = st["verb"], _names(st, names), _where(st)
    loc = f" in the {where}" if where else ""
    d = st["done_when"]
    cue = d.get("raw") or (f"until {d['target']}" if d.get("target") else None)
    if v == "heat" and d["kind"] in ("visual", "texture", "temp") and cue:
        return f"the {where or 'pan'} is on a burner that is on, {cue}"
    if d["kind"] in ("visual", "texture", "temp") and cue:
        return f"the {what}{loc} {_be(what)} {PAST.get(v, v + 'ed')}, {cue}"
    where = where or "its container"
    if v == "heat":
        return f"the {where} is on a burner that is on" + (f" ({st['settings']['heat']} heat)" if st["settings"].get("heat") else "")
    if v == "preheat":
        return f"the oven is on at {st['settings'].get('temp_c', '?')} C"
    if v == "turn_off":
        return "every burner is off"
    if v == "transfer":
        return f"the {what} {_be(what)} in the {where}"
    if v in ("mix", "season"):
        return f"the {what} {_be(what)} mixed in the {where}"
    if v in ("cook", "bake", "melt"):
        return f"the {what} in the {where} {_be(what)} {PAST[v]}"
    if v == "crack":
        return f"the {what} {_be(what)} cracked into the {where}, with no shell"
    if v == "beat":
        return f"the {what} in the {where} {_be(what)} beaten"
    if v == "cut":
        return f"the {what} is cut" + (f" ({st['style']})" if st.get("style") else "")
    if v in ("wait", "cool"):
        return f"the {what} has rested" + (f" {int(st['settings']['duration_s'])} s" if st["settings"].get("duration_s") else "")
    if v in ("cover", "uncover"):
        return f"the {where} is {'covered' if v == 'cover' else 'uncovered'}"
    return f"{v} done: {what}"


def compile_job(actions: dict, name: str | None = None, include_unsupported: bool = True) -> dict[str, Any]:
    ins, feas = actions["instructions"], actions.get("feasibility") or {}
    status = {p["id"]: p for p in feas.get("per_step", [])}
    names = {i["id"]: i["name"] for i in ins.get("inventory", {}).get("ingredients", [])}
    main = [s for s in ins["steps"] if s["on_main_path"]]
    keep = [s for s in main if include_unsupported or status.get(s["id"], {}).get("status") != "unsupported"]
    ids = {s["id"] for s in keep}
    after = {s["id"]: s["after"] for s in ins["steps"]}

    def deps_of(sid: str, seen=None) -> list[str]:
        """Dependencies among kept steps; a dropped step passes its own dependencies on."""
        out, seen = [], seen or set()
        for d in after.get(sid, []):
            if d in seen:
                continue
            seen.add(d)
            out += [d] if d in ids else deps_of(d, seen)
        return sorted(set(out))
    steps, human = [], []
    for s in keep:
        hint = []
        if s["settings"].get("duration_s"):
            hint.append(f"about {int(s['settings']['duration_s'])} s")
        if s["done_when"]["kind"] in ("visual", "texture") and s["done_when"].get("raw"):
            hint.append(f"recipe says: {s['done_when']['raw']}")
        hint += [f"{r['kind']}: {r['text']}" for r in s.get("rules", [])]
        hint += [f"open question: {q}" for q in s.get("open_questions", [])]
        f = status.get(s["id"])
        if f and f["status"] != "supported":
            hint.append(f"ROBOT {f['status'].upper()}: {f.get('note') or 'no skill'}")
            if f["status"] == "unsupported":
                human.append(s["id"])
        if s["verb"] in ("heat", "preheat", "turn_off"):
            hint.insert(0, f"recipe says: {s['instruction']}")
        step = {"id": s["id"], "goal": _goal(s, names), "done_when": _effect(s, names)}
        deps = deps_of(s["id"])
        if deps:
            step["depends_on"] = deps
        if hint:
            step["hint"] = "; ".join(dict.fromkeys(h for h in hint
                                                   if not (h.startswith("recipe says: until") and
                                                           any(x.startswith("recipe says: ") and h[13:] in x and x != h for x in hint))))
        steps.append(step)
    title = ins.get("title") or "recipe"
    return {
        "name": name or re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_"),
        "goal": f"{title}: " + (", ".join(x.replace("_", " ") for x in ins.get("final_items", [])) or "done"),
        "steps": steps,
        "source": {"compiled_from": "recipe_planning actions.json", "recipe_status": ins.get("status"),
                   "skills_profile": feas.get("profile"), "supported": feas.get("supported"),
                   "partial": feas.get("partial"), "unsupported": feas.get("unsupported"),
                   "needs_a_person": human, "open_questions": ins.get("open_questions", [])},
    }
