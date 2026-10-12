"""Wire protocol for the agent <-> robot hop: msgpack-numpy with tagged dataclasses.

One Request per call, one Response per Request, matched by `id`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import msgpack
import msgpack_numpy

from chefbot.interface.types import RGBD, Detection, EnvState, MotionResult, Pose2D, ToolResult

ROBOT = "robot"
DETECTOR = "detector"

# Methods a client may invoke, per target. Anything else is rejected.
ALLOWED_METHODS: dict[str, frozenset[str]] = {
    ROBOT: frozenset(
        {"go_to_pose", "align_to_point", "wait", "get_status", "stop_base", "get_base_pose"}
    ),
    DETECTOR: frozenset({"detect"}),
}


@dataclass
class Request:
    id: int
    target: str  # ROBOT | DETECTOR
    method: str
    args: list[Any] = field(default_factory=list)


@dataclass
class Response:
    id: int
    ok: bool
    result: Any = None
    error: str | None = None


_TYPE_KEY = "__chefbot__"
_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (Pose2D, Detection, MotionResult, EnvState, RGBD, ToolResult, Request, Response)
}


def _encode(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and type(obj).__name__ in _TYPES:
        out = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        out[_TYPE_KEY] = type(obj).__name__
        return out
    return msgpack_numpy.encode(obj)


def _decode(obj: dict) -> Any:
    obj = msgpack_numpy.decode(obj)
    if isinstance(obj, dict) and _TYPE_KEY in obj:
        cls = _TYPES[obj.pop(_TYPE_KEY)]
        return cls(**obj)
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpack(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode, raw=False)
