"""The onboard config, and the drivers and cameras built from it (no server: check.py uses these alone)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from chefbot.onboard.cameras import Cameras
from chefbot.onboard.frames import BaseFrames
from chefbot.onboard.mocap import MocapFreshness
from chefbot.onboard.robot_io import RedisRobot

CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_robot(config: dict, redis_client=None) -> RedisRobot:
    if redis_client is None:
        import redis

        redis_client = redis.Redis(host=config["redis"]["host"], port=int(config["redis"]["port"]), decode_responses=True)
        redis_client.ping()
    base = config["base"]
    frames = BaseFrames(np.radians(base["marker_yaw_offset_deg"]), base["marker_xy"])
    mocap = MocapFreshness(base.get("mocap_stale_s", 0.25))
    return RedisRobot(redis_client, frames, config["gripper"], (base["driver_max_vel_xy"], base["driver_max_vel_yaw"]), mocap)


def build_cameras(config: dict) -> Cameras:
    return Cameras(config.get("cameras") or [], config["server"]["camera_idle_s"])
