"""RoboCasa backend configuration (config/envs/robocasa.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_PATH = PACKAGE_ROOT / "config" / "envs" / "robocasa.yaml"


@dataclass(frozen=True)
class RoboCasaConfig:
    task: str = "Kitchen"
    robot: str = "PandaOmron"
    layout: int = 1
    style: int = 1
    seed: int = 0
    control_freq: int = 20
    physics_timestep: float = 0.002
    shoulder_camera: str = "robot0_frontview"
    camera_width: int = 640
    camera_height: int = 480
    grid_resolution: float = 0.05
    grid_height_band: tuple[float, float] = (0.05, 1.8)
    grid_margin: float = 0.5
    maps_dir: str = "maps"
    server_host: str = "127.0.0.1"
    server_port: int = 8765

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "RoboCasaConfig":
        raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text())
        config = cls(
            task=raw["task"],
            robot=raw["robot"],
            layout=raw["layout"],
            style=raw["style"],
            seed=raw["seed"],
            control_freq=raw["control_freq"],
            physics_timestep=raw["physics_timestep"],
            shoulder_camera=raw["shoulder_camera"],
            camera_width=raw["camera"]["width"],
            camera_height=raw["camera"]["height"],
            grid_resolution=raw["grid"]["resolution"],
            grid_height_band=tuple(raw["grid"]["height_band"]),
            grid_margin=raw["grid"]["margin"],
            maps_dir=raw["maps_dir"],
            server_host=raw["server"]["host"],
            server_port=raw["server"]["port"],
        )
        return replace(config, **{k: v for k, v in overrides.items() if v is not None})

    @property
    def map_dir(self) -> Path:
        """Where this layout/style's semantic map and grid live."""
        base = Path(self.maps_dir)
        base = base if base.is_absolute() else REPO_ROOT / base
        return base / f"robocasa_layout{self.layout:03d}_style{self.style:03d}"
