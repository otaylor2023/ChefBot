"""Motion thread: one motion at a time, preemptible, with a queryable result."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from chefbot.interface.types import MotionResult, MotionStatus, Pose2D

# A motion body: runs until done or until the preempt event is set.
MotionFn = Callable[[threading.Event], MotionResult]


@dataclass
class _Motion:
    thread: threading.Thread | None = None
    preempt: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: MotionResult | None = None


class MotionManager:
    def __init__(self, get_pose: Callable[[], Pose2D]):
        self._get_pose = get_pose
        self._lock = threading.Lock()  # serializes start/stop
        self._current: _Motion | None = None

    def start(self, run: MotionFn) -> None:
        """Start a motion on its own thread, preempting any motion in progress."""
        with self._lock:
            self._preempt(self._current)
            motion = _Motion()
            motion.thread = threading.Thread(
                target=self._run, args=(motion, run), name="motion", daemon=True
            )
            self._current = motion
            motion.thread.start()

    def stop(self) -> bool:
        """Preempt the motion in progress and wait for it to exit.
        Returns False if nothing was moving."""
        with self._lock:
            return self._preempt(self._current)

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        motion = self._current
        if motion is None:
            return self._idle()
        motion.done.wait(timeout_s)
        return self._snapshot(motion)

    def status(self) -> MotionResult:
        motion = self._current
        return self._idle() if motion is None else self._snapshot(motion)

    def _idle(self) -> MotionResult:
        return MotionResult(MotionStatus.IDLE, self._get_pose(), "no motion has been started")

    def _snapshot(self, motion: _Motion) -> MotionResult:
        if motion.done.is_set() and motion.result is not None:
            return motion.result
        return MotionResult(MotionStatus.MOVING, self._get_pose(), "motion in progress")

    @staticmethod
    def _preempt(motion: _Motion | None) -> bool:
        if motion is None or motion.done.is_set():
            return False
        motion.preempt.set()
        if motion.thread is not None:
            motion.thread.join()
        return True

    def _run(self, motion: _Motion, run: MotionFn) -> None:
        try:
            motion.result = run(motion.preempt)
        except Exception as exc:  # a crashed motion must still report a result
            motion.result = MotionResult(
                MotionStatus.ERROR, self._safe_pose(), f"{type(exc).__name__}: {exc}"
            )
        finally:
            motion.done.set()

    def _safe_pose(self) -> Pose2D | None:
        try:
            return self._get_pose()
        except Exception:
            return None
