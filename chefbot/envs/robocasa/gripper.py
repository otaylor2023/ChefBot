"""Robotiq 2F-85 (MuJoCo Menagerie model) as a robosuite gripper.

robosuite ships its own `Robotiq85Gripper`, but that model's fingers jam on
their own linkage at a ~6 cm gap and go unstable on repeated closes, so it
cannot hold small objects. Menagerie's model has the real closed 4-bar linkage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel

GRIPPER_XML = Path(__file__).resolve().parent / "assets" / "robotiq_2f85" / "gripper.xml"
GRIPPER_NAME = "Menagerie2F85Gripper"
# Collision geoms of the two fingertip pads, in the gripper's own (unprefixed) names.
PAD_GEOMS = {
    "left": ("left_pad1_collision", "left_pad2_collision"),
    "right": ("right_pad1_collision", "right_pad2_collision"),
}


class Menagerie2F85Gripper(GripperModel):
    """One action: -1 opens, +1 closes."""

    def __init__(self, idn=0):
        super().__init__(str(GRIPPER_XML), idn=idn)

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.25

    @property
    def dof(self):
        return 1

    @property
    def init_qpos(self):
        # driver, coupler, spring_link, follower for each side: all zero is fully open.
        return np.zeros(8)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_follower_collision", *PAD_GEOMS["left"]],
            "right_finger": ["right_follower_collision", *PAD_GEOMS["right"]],
            "left_fingerpad": list(PAD_GEOMS["left"]),
            "right_fingerpad": list(PAD_GEOMS["right"]),
        }


def register_gripper() -> str:
    """Make the gripper available to `robosuite.make(gripper_types=...)`."""
    GRIPPER_MAPPING.setdefault(GRIPPER_NAME, Menagerie2F85Gripper)
    return GRIPPER_NAME
