"""Seeded cup scenes (spec §4): where the cup goes, which cup, which distractors.

A scene is pure data (JSON-able): built once per config's scene set and saved,
so every config and repeat sees exactly the same scenes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Scene:
    index: int
    seed: int
    cup_model: str
    cup_xy: tuple[float, float]  # robot frame (m)
    cup_yaw: float  # rad
    distractors: list[dict] = field(default_factory=list)  # {"key", "xy", "yaw"}

    @property
    def name(self) -> str:
        return f"s{self.index:02d}"


def generate(config: dict, reachable: Callable[[np.ndarray], bool]) -> list[Scene]:
    """`config["scenes"]` scenes from `config["scene_seed"]`. `reachable(xy)`
    says whether a top-down grasp at a cup there has an IK solution."""
    spec = config["scene_sampling"]
    root = np.random.default_rng(int(config["scene_seed"]))
    scenes = []
    for index in range(int(config["scenes"])):
        seed = int(root.integers(1 << 31))
        rng = np.random.default_rng(seed)
        cup = config["cup_models"][index % len(config["cup_models"])]
        for _ in range(200):
            xy = np.array([rng.uniform(*spec["forward"]), rng.uniform(*spec["sideways"])])
            if reachable(xy):
                break
        else:
            raise RuntimeError(f"scene {index}: no reachable cup position in 200 tries")
        distractors = []
        if index % 2 == 1:  # distractors on in every other scene
            keys = [d["category"] for d in config["distractors"]]
            count = int(rng.integers(spec["distractors"][0], spec["distractors"][1] + 1))
            placed = [xy]
            for key in rng.permutation(keys)[:count]:
                for _ in range(100):
                    angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(spec["distractor_min"], spec["distractor_max"])
                    p = xy + dist * np.array([np.cos(angle), np.sin(angle)])
                    if all(np.linalg.norm(p - q) >= spec["distractor_min"] for q in placed) and spec["counter_x"][0] <= p[0] <= spec["counter_x"][1] and spec["counter_y"][0] <= p[1] <= spec["counter_y"][1]:
                        placed.append(p)
                        distractors.append({"key": str(key), "xy": [float(p[0]), float(p[1])], "yaw": float(rng.uniform(-np.pi, np.pi))})
                        break
        scenes.append(Scene(index, seed, cup, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi)), distractors))
    return scenes


def save(scenes: list[Scene], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in scenes], indent=1))


def load(path: Path) -> list[Scene]:
    return [Scene(**{**d, "cup_xy": tuple(d["cup_xy"])}) for d in json.loads(path.read_text())]
