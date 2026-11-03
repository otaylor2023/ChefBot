"""How fresh the mocap publisher's pose is.

The publisher (bringup/mocap.sh: NatNet's SampleClient) writes `<body>::pos`, `::ori` and
`::tracking_valid` every frame, with no frame number or timestamp. When it stops, its last values
stay in Redis, `tracking_valid` included, and they survive a reboot (the robot's Redis saves to
disk). So a pose counts as fresh only if its raw value changed within `stale_after_s`, and a value
that was already there when we started reading is never fresh until it changes: a dead publisher
reads as stale instead of as a valid pose frozen in place.

At rest the pose still changes every frame (Motive's sub-millimetre noise in the publisher's six
digits); if a still robot ever reads stale, that assumption is what failed.
"""

from __future__ import annotations

import time
from typing import Callable


class MocapFreshness:
    def __init__(self, stale_after_s: float = 0.25, clock: Callable[[], float] = time.monotonic):
        self.stale_after_s = float(stale_after_s)
        self._clock = clock
        self._last: tuple | None = None  # the raw (position, orientation) last seen
        self._changed_at: float | None = None  # when it last changed; None: never seen a change

    def update(self, raw_position, raw_orientation) -> float | None:
        """Note the values just read; return the seconds since they last changed (None: not since we started)."""
        now = self._clock()
        current = (raw_position, raw_orientation)
        if self._last is not None and current != self._last and raw_position is not None:
            self._changed_at = now
        self._last = current
        return None if self._changed_at is None else now - self._changed_at

    def is_fresh(self, age_s: float | None) -> bool:
        return age_s is not None and age_s <= self.stale_after_s


Freshness = MocapFreshness  # the same test for any value a live publisher keeps rewriting (the base driver's heartbeat)
