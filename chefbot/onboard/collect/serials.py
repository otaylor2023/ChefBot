"""The RealSenses' serials by camera name, from onboard/config.yaml `cameras:` (their one copy, the server's too), for
the recorders next to this file, which run as plain files in the robot's camera env and import nothing from chefbot."""

from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def serials() -> dict[str, str]:
    return {c["name"]: str(c["serial"]) for c in yaml.safe_load(CONFIG.read_text())["cameras"]}
