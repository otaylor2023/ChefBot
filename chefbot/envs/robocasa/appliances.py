"""Burners: where they are and what turns them on (the stove's knob joints).

Burners are numbered burner_1, burner_2, ... per stove in RoboCasa's location
order (front left, front right, ...): the same names on the semantic map, in
the observer's text, and for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robocasa.models.fixtures.stove import STOVE_LOCATIONS, Stove

from chefbot.envs.robocasa.env import RoboCasaEnv


@dataclass(frozen=True)
class Burner:
    name: str  # "burner_1"
    stove: str  # RoboCasa fixture name
    location: str  # "front_left", ...
    site: str  # the burner's place site


def burners(env: RoboCasaEnv) -> list[Burner]:
    out = []
    for sim_name, fixture in sorted(env.suite.fixtures.items()):
        if not isinstance(fixture, Stove):
            continue
        for location in STOVE_LOCATIONS:
            site = f"{fixture.naming_prefix}burner_{location}_place_site"
            if fixture.knob_joints.get(location) is None or _site_id(env, site) < 0:
                continue
            out.append(Burner(f"burner_{len(out) + 1}", sim_name, location, site))
    return out


def burner_position(env: RoboCasaEnv, burner: Burner) -> np.ndarray:
    return env.data.site_xpos[_site_id(env, burner.site)].copy()


def burner_on(env: RoboCasaEnv, burner: Burner) -> bool:
    return bool(env.suite.fixtures[burner.stove].is_burner_on(env.suite, burner.location))


def set_burner(env: RoboCasaEnv, burner: Burner, on: bool, rng: np.random.RandomState) -> None:
    stove = env.suite.fixtures[burner.stove]
    stove.set_knob_state(env.suite, rng, knob=burner.location, mode="on" if on else "off")
    stove.update_state(env.suite)  # the flame


def nearest_burner(env: RoboCasaEnv, position, radius: float = 0.15) -> Burner | None:
    position = np.asarray(position, dtype=float)
    best, best_d = None, radius
    for burner in burners(env):
        d = float(np.linalg.norm(burner_position(env, burner)[:2] - position[:2]))
        if d <= best_d:
            best, best_d = burner, d
    return best


def _site_id(env: RoboCasaEnv, name: str) -> int:
    import mujoco

    return int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name))
