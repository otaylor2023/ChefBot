import dataclasses

import numpy as np
import pytest

from chefbot.interface.transport.protocol import Request, Response, pack, unpack
from chefbot.interface.types import (
    RGBD,
    ActuationResult,
    Detection,
    EnvState,
    Grasp,
    HeldObject,
    MotionResult,
    Placement,
    Pose2D,
    ToolResult,
)


def same(a, b) -> bool:
    """Deep equality that understands numpy arrays and dataclasses."""
    if dataclasses.is_dataclass(a):
        return type(a) is type(b) and all(
            same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


POSE = Pose2D(1.5, -2.25, 0.75)
DETECTION = Detection("fridge handle", np.array([0.6, 2.1, 1.05]), 0.87)
MESSAGES = [
    POSE,
    DETECTION,
    Detection("counter", np.array([3.0, 0.3, 0.92]), 1.0, near_point_world=np.array([3.0, 0.5, 0.92])),
    MotionResult("success", POSE, "the fridge is now open (scripted)", {"scripted": True}),
    EnvState(
        POSE, np.zeros(3), False, 3.0, arm_q=np.linspace(-1, 1, 7), arm_qd=np.zeros(7),
        gripper_width=0.045, gripper_closed_cmd=True, arm_in_contact=True, ee_wrench=np.arange(6.0),
    ),  # fmt: skip
    Grasp(np.eye(4), 0.1, 0.045, 0.9, retreat=np.array([-0.16, 0.0, 0.02])),
    Grasp(np.eye(4), 0.1, 0.045),
    Placement(np.eye(4), 0.1, 0.8),
    HeldObject("egg", 0.045, np.array([4.5, -0.6, 1.39])),
    ActuationResult(True, True, "open", "fridge_main_group", np.array([[4.9, -1.0], [4.9, -1.1]])),
    ActuationResult(False, True, "egg is not an openable fixture"),
    Request(7, "robot", "pick", [DETECTION]),
    Request(8, "robot", "actuate_fixture", [DETECTION, True]),
    Response(9, True, HeldObject("mug", 0.07, np.zeros(3))),
    MotionResult("success", POSE, "reached the goal"),
    MotionResult("idle", None, ""),
    EnvState(POSE, np.array([0.1, -0.2, 0.3]), True, 12.5),
    RGBD(
        rgb=np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        depth=np.linspace(0.5, 3.0, 6, dtype=np.float32).reshape(2, 3),
        intrinsics=np.array([[500.0, 0, 1.5], [0, 500.0, 1.0], [0, 0, 1.0]]),
        cam_to_world=np.eye(4),
    ),
    ToolResult(True, None, "Found: burner_1.", {"detections": [{"id": "burner_1", "label": "burner"}]}),
    ToolResult(False, "too_far", "Use go_to first.", {"pose": {"x": 1.0, "y": 2.0, "yaw_deg": 90.0}}),
    Request(0, "robot", "go_to_pose", [POSE]),
    Request(1, "robot", "align_to_point", [np.array([3.4, 1.0, 0.9])]),
    Request(2, "robot", "wait", [None]),
    Request(3, "robot", "wait", [2.5]),
    Request(4, "robot", "get_status", []),
    Request(5, "detector", "detect", [["burner", "pan"]]),
    Response(0, True, None, None),
    Response(1, True, POSE, None),
    Response(2, True, MotionResult("moving", POSE, "motion in progress"), None),
    Response(3, True, [DETECTION, Detection("pan", np.zeros(3), 1.0)], None),
    Response(4, True, [], None),
    Response(5, False, None, "RuntimeError: boom"),
]


@pytest.mark.parametrize("message", MESSAGES, ids=lambda m: type(m).__name__)
def test_round_trip(message):
    decoded = unpack(pack(message))
    assert same(decoded, message), decoded


def test_arrays_keep_dtype_and_shape():
    rgbd = unpack(pack(next(m for m in MESSAGES if isinstance(m, RGBD))))
    assert rgbd.rgb.dtype == np.uint8 and rgbd.rgb.shape == (2, 3, 3)
    assert rgbd.depth.dtype == np.float32


def test_pose_stays_a_pose_inside_containers():
    decoded = unpack(pack({"poses": [POSE, POSE], "n": 2}))
    assert decoded == {"poses": [POSE, POSE], "n": 2}
    assert isinstance(decoded["poses"][0], Pose2D)
