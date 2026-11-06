"""The base's hard speed cap and the speed profiles under it: one copy, read as data by the laptop (embodiments' Robot.speed)
and the robot computer (the onboard server, launch.sh's base driver, the gamepad).

The cap is the embodiment's robot.yaml `control.limits`: base_max_vel [x, y, yaw] (m/s, m/s, rad/s) and base_max_accel
(m/s^2, m/s^2, rad/s^2). The driver runs at it; nothing that moves the base may ask for more: a profile whose base
speed is over it is refused, not clipped (a run asking for more than the robot allows is a mistake to say, not to
paper over).

The profiles are embodiments/speed_profiles.yaml. A profile may set the arm's rates, the base's, or both; a run picks one
for the arm and one for the base (run_real --arm-speed / --base-speed; --speed both). `teleop` is the gamepad's (the
base's only).

    cap = base_cap("tidybot_fr3")                     # {"vel": (vx, vy, vyaw), "accel": (ax, ay, ayaw)}
    vel, accel = base_profile("teleop", "tidybot_fr3")  # refused (ValueError) if over the cap
"""

from __future__ import annotations

from pathlib import Path

import yaml

EMBODIMENTS = Path(__file__).resolve().parent.parent / "embodiments"
PROFILES = EMBODIMENTS / "speed_profiles.yaml"
BASE_KEYS = ("base_max_vel",)
ARM_KEYS = ("arm_lin", "arm_ang", "joint")


def profiles() -> dict:
    return yaml.safe_load(PROFILES.read_text())


def profile(name: str) -> dict:
    ps = profiles()
    if name not in ps:
        raise ValueError(f"no speed profile {name!r}; profiles: {', '.join(ps)}")
    return ps[name]


def base_cap(embodiment: str = "tidybot_fr3") -> dict:
    """The embodiment's hard base cap: {"vel": (x, y, yaw), "accel": (x, y, yaw) or None}."""
    raw = yaml.safe_load((EMBODIMENTS / embodiment / "robot.yaml").read_text())
    lim = (raw.get("control") or {}).get("limits") or {}
    if "base_max_vel" not in lim:
        raise ValueError(f"embodiment {embodiment!r} has no control.limits.base_max_vel (its base's hard cap)")
    accel = lim.get("base_max_accel")
    return {"vel": tuple(float(v) for v in lim["base_max_vel"]), "accel": None if accel is None else tuple(float(v) for v in accel)}


def base_stop(embodiment: str = "tidybot_fr3") -> dict:
    """The base's measured stop (control.limits base_stop_tau_s, base_stop_brake_mps2), {} if the embodiment has none
    (navigation then refuses to size its speed by it: robot/nav/speed.py StopModel.from_limits)."""
    raw = yaml.safe_load((EMBODIMENTS / embodiment / "robot.yaml").read_text())
    lim = (raw.get("control") or {}).get("limits") or {}
    return {k: float(lim[k]) for k in ("base_stop_tau_s", "base_stop_brake_mps2") if k in lim}


def base_profile(name: str, embodiment: str = "tidybot_fr3") -> tuple[tuple, tuple | None]:
    """A profile's base velocity and acceleration (the cap's acceleration if the profile sets none), refused if either is
    over the embodiment's cap or the profile has no base speed."""
    p, cap = profile(name), base_cap(embodiment)
    if "base_max_vel" not in p:
        raise ValueError(f"speed profile {name!r} has no base speed (base_max_vel)")
    vel = tuple(float(v) for v in p["base_max_vel"])
    accel = tuple(float(v) for v in p["base_max_accel"]) if "base_max_accel" in p else cap["accel"]
    over = [f"{what} {mine} > {limit}" for what, mine, limit in (("velocity", vel, cap["vel"]), ("acceleration", accel, cap["accel"]))
            if mine is not None and limit is not None and any(a > b + 1e-9 for a, b in zip(mine, limit))]  # fmt: skip
    if over:
        raise ValueError(f"speed profile {name!r} is over {embodiment}'s base cap ({'; '.join(over)}): pick a slower one")
    return vel, accel


def arm_profile(name: str) -> tuple[float, float, float]:
    """A profile's arm rates (arm_lin m/s, arm_ang rad/s, joint rad/s), refused if it has none."""
    p = profile(name)
    missing = [k for k in ARM_KEYS if k not in p]
    if missing:
        raise ValueError(f"speed profile {name!r} has no arm rates ({', '.join(missing)}): it is for the base only")
    return float(p["arm_lin"]), float(p["arm_ang"]), float(p["joint"])
