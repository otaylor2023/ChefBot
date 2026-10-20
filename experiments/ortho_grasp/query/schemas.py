"""Prompt text and output schemas per config (spec §8, §10, §11), and parsing
a reply into marks (plot coordinates per view) with a failure reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from experiments.ortho_grasp.render.plots import PlotSpec

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


class ParseError(ValueError):
    pass


def names(cfg: dict) -> list[str]:
    return ["A", "B", "F"] if cfg["representation"] == "p1" else ["A", "B"]


def _task_long(cfg: dict) -> str:
    return "pick up the cup" if cfg["trajectory"] == "t3" else "grasp the cup"


def _appearance(cfg: dict) -> str:
    if cfg["render"] == "v4":
        return "The object (a cup) is shown as a rendered picture of it; everything else around it is removed."
    seen = (" as seen by the robot's camera (only the side facing the camera is visible, so parts of it are missing)"
            if cfg["geometry"] == "partial" else "")  # fmt: skip
    return (f"The object (a cup) is drawn as a cloud of points on its surface{seen}, colored by height "
            "(dark purple: low, at the floor of the cup; yellow: high, at the rim; see the color bar).")  # fmt: skip


def _marks_text(cfg: dict) -> str:
    if cfg["representation"] == "p1":
        return ("Mark three points: A and B, where the two finger pads should be once the fingers close, with the part "
                "of the cup you hold between them; and F, the center of the gripper's flange (the red bar at the top "
                "of the gripper in the panel), which is 15.6 cm behind the midpoint of A and B. F sets the direction the "
                "gripper comes from: the gripper's body lies along the line from the midpoint of A and B to F, and must "
                "not pass through the cup or the counter.")  # fmt: skip
    return ("Mark two points: A and B, where the two finger pads should be once the fingers close, with the part of "
            "the cup you hold between them. The direction the gripper comes from is chosen for you.")  # fmt: skip


def _point_fmt(cfg: dict, view: str, n: list[str]) -> str:
    if cfg["output"] == "o2":
        return "{" + ", ".join(f'"{k}": [u, v]' for k in n) + "}"
    second = "y" if view == "top" else "z"
    return "{" + ", ".join(f'"{k}": [x, {second}]' for k in n) + "}"


def _units(cfg: dict, size: int) -> str:
    if cfg["output"] == "o2":
        return f"Each value is a pixel position in that image: u to the right, v down, from the top-left corner ({size} x {size} px)."
    return "Each value is in meters, read off that plot's axes."


def prompt(cfg: dict, size: int) -> str:
    """The single prompt (C-ind) or the first call (C-seq, top view only)."""
    n = names(cfg)
    if cfg["representation"] == "b0":
        raise ValueError("use prompt_b0")
    template = (PROMPTS / cfg["prompt_version"] / "plots.md").read_text()
    seq = cfg["views"] == "seq"
    if seq and cfg["trajectory"] == "t3":
        raise ValueError("sequential views are only defined for t1")
    views = "One plot is attached: the TOP VIEW of the object. (You will give the heights in a second step.)" if seq else \
        "Two plots of the object are attached: first the TOP VIEW, then the SIDE VIEW."  # fmt: skip
    pts = ", ".join(n)
    if cfg["trajectory"] == "t3":
        task = ("Pick up the cup. Draw the gripper at three waypoints along the motion: 1 = pre-grasp (open, about 10 cm "
                "back from the grasp, along the direction it comes from), 2 = grasp (fingers around the cup, about to "
                f"close), 3 = lift (closed, with the cup lifted clear of the counter). For each waypoint mark {pts} in both "
                "views and say whether the gripper is open or closed there. A point must be the same point in both views: "
                "its x should match.")  # fmt: skip
        wp = "{" + f'"gripper": "open" or "closed", "top": {_point_fmt(cfg, "top", n)}, "side": {_point_fmt(cfg, "side", n)}' + "}"
        fmt = "```json\n{\"waypoints\": [" + wp + ", (waypoint 2), (waypoint 3)]}\n```"
    else:
        task = "Grasp the cup: put the fingers around part of it, ready to close."
        if seq:
            task += f" Mark {pts} in the top view."
            fmt = "```json\n{\"top\": " + _point_fmt(cfg, "top", n) + "}\n```"
        else:
            task += f" Mark {pts} in both views. A point must be the same point in both views: its x should match."
            fmt = "```json\n{\"top\": " + _point_fmt(cfg, "top", n) + ", \"side\": " + _point_fmt(cfg, "side", n) + "}\n```"
    fmt += "\n" + _units(cfg, size)
    return template.format(task_long=_task_long(cfg), views_text=views, appearance_text=_appearance(cfg),
                           marks_text=_marks_text(cfg), task_text=task, format_text=fmt)  # fmt: skip


def prompt_seq_side(cfg: dict, size: int, previous: dict, guides: dict[str, float]) -> str:
    template = (PROMPTS / cfg["prompt_version"] / "seq_side.md").read_text()
    n = names(cfg)
    guides_text = "\n".join(f"- {k}: x = {x:.3f} m" for k, x in guides.items())
    if cfg["output"] == "o2":
        fmt = "```json\n{\"side\": {" + ", ".join(f'"{k}": v' for k in n) + "}}\n```\n" \
              f"Each value is the pixel row v of that point on its line (v down from the top edge; the image is {size} x {size} px)."  # fmt: skip
    else:
        fmt = "```json\n{\"side\": {" + ", ".join(f'"{k}": z' for k in n) + "}}\n```\nEach value is the height z in meters, read off the plot's axis."
    return template.format(appearance_text=_appearance(cfg), guides_text=guides_text, previous=json.dumps(previous),
                           task_long=_task_long(cfg), marks_text=_marks_text(cfg), format_text=fmt)  # fmt: skip


def prompt_b0(cfg: dict, camera_text: str, counter_z: float, tcp_position, tcp_quaternion) -> str:
    template = (PROMPTS / cfg["prompt_version"] / "b0.md").read_text()
    fmt = ("```json\n{\"position\": [x, y, z], \"quaternion_wxyz\": [w, x, y, z], \"width\": w}\n```\n"
           "position: the TCP at the grasp, in meters (robot frame). quaternion_wxyz: the TCP orientation at the grasp "
           "(robot frame, unit quaternion, w first). width: the distance between the finger pads at the grasp, before "
           "closing, in meters (at most 0.085).")  # fmt: skip
    task = "Grasp the cup: give the gripper pose at which its fingers are around part of the cup, ready to close."
    return template.format(task_long=_task_long(cfg), camera_text=camera_text, counter_z=counter_z,
                           tcp_position=[round(float(v), 4) for v in tcp_position],
                           tcp_quaternion=[round(float(v), 4) for v in tcp_quaternion], task_text=task, format_text=fmt)  # fmt: skip


# -- parsing -------------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = blocks[::-1] or [m.group(0) for m in re.finditer(r"\{.*\}", text, re.S)][::-1]
    for c in candidates:
        try:
            out = json.loads(c)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    raise ParseError("no_json")


def _pair(v, what: str) -> np.ndarray:
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
        raise ParseError(f"bad_value:{what}")
    return np.array(v, dtype=float)


def _view_marks(cfg: dict, d: dict, spec: PlotSpec, n: list[str], where: str) -> dict[str, np.ndarray]:
    if not isinstance(d, dict):
        raise ParseError(f"missing:{where}")
    out = {}
    for k in n:
        if k not in d:
            raise ParseError(f"missing:{where}.{k}")
        v = _pair(d[k], f"{where}.{k}")
        hv = spec.to_world(v)[0] if cfg["output"] == "o2" else v
        if not spec.inside(hv)[0]:
            raise ParseError(f"out_of_range:{where}.{k}")
        out[k] = hv
    return out


def parse_marks(cfg: dict, text: str, specs: dict[str, PlotSpec]) -> dict:
    """C-ind (and the first C-seq call): {"top": {...}, "side": {...}} or
    {"waypoints": [{"gripper", "top", "side"}]}, as plot coordinates."""
    d = extract_json(text)
    n = names(cfg)
    if cfg["trajectory"] == "t3":
        wps = d.get("waypoints")
        if not isinstance(wps, list) or len(wps) != 3:
            raise ParseError("bad_waypoints")
        out = []
        for i, wp in enumerate(wps):
            g = wp.get("gripper") if isinstance(wp, dict) else None
            if g not in ("open", "closed"):
                raise ParseError(f"bad_value:waypoints[{i}].gripper")
            out.append({"gripper": g, "top": _view_marks(cfg, wp.get("top"), specs["top"], n, f"wp{i + 1}.top"),
                        "side": _view_marks(cfg, wp.get("side"), specs["side"], n, f"wp{i + 1}.side")})  # fmt: skip
        return {"waypoints": out}
    out = {"top": _view_marks(cfg, d.get("top"), specs["top"], n, "top")}
    if cfg["views"] == "ind":
        out["side"] = _view_marks(cfg, d.get("side"), specs["side"], n, "side")
    return out


def parse_seq_side(cfg: dict, text: str, spec: PlotSpec) -> dict[str, float]:
    d = extract_json(text).get("side")
    if not isinstance(d, dict):
        raise ParseError("missing:side")
    out = {}
    for k in names(cfg):
        v = d.get(k)
        if not isinstance(v, (int, float)):
            raise ParseError(f"bad_value:side.{k}")
        z = float(spec.to_world([[spec.left, v]])[0][1]) if cfg["output"] == "o2" else float(v)
        if not spec.v_range[0] <= z <= spec.v_range[1]:
            raise ParseError(f"out_of_range:side.{k}")
        out[k] = z
    return out


def parse_b0(text: str) -> tuple[np.ndarray, np.ndarray, float]:
    d = extract_json(text)
    try:
        p = np.array(d["position"], dtype=float)
        q = np.array(d["quaternion_wxyz"], dtype=float)
        w = float(d["width"])
    except (KeyError, TypeError, ValueError):
        raise ParseError("bad_b0")
    if p.shape != (3,) or q.shape != (4,) or np.linalg.norm(q) < 1e-6 or not np.all(np.isfinite(p)):
        raise ParseError("bad_b0")
    return p, q, w
