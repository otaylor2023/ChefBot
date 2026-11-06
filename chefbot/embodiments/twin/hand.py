"""The Franka Hand as the real ChefBot drives it: libfranka's two gripper commands, through OpenSai's gripper driver
(github.com/manips-sai/FrankaPanda, redis_driver/gripper.cpp; chefbot/onboard/robot_io.py writes its keys).

    move(width, speed)   the fingers to `width` (m between them) at `speed` (m/s), position-controlled and stiff,
                         pushing as hard as it must, up to the hand's 70 N. If something stops them short, the move
                         ends there and the hand keeps pushing toward the width (it holds what is between the fingers
                         as hard as it can). (libfranka does not document a blocked move; OpenSai's driver retries 1 cm
                         wider if libfranka reports one failed. Assumed here, unmeasured: it keeps pushing.)
    grasp(speed, force)  close at `speed` until the fingers stop on something (or meet), then hold `force` (N) on it:
                         the driver's grasp(0, speed, force, 1.0, 1.0), any width counts as grasped.

The driver blocks on each call: a command sent while one runs waits for it to finish, and a later one replaces the
waiting one (the driver reads the latest keys). The force is the hand's own current control: nothing measures it, and
the skill only gets the fingers' width back.

In the sim the finger tendon's position servo follows a reference that moves at the commanded speed, its closing force
limited to the command's (a grasp's force, or the hand's 70 N in a move), stiff in a move (light resistance such as a
tool's spring doesn't make it lag); once a grasp's fingers stop on something, the servo gives way to a constant squeeze
of exactly that force.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_WIDTH = 0.08  # m between the fingers (libfranka's max_width)
MAX_SPEED = 0.1  # m/s of width (50 mm/s per finger)
MAX_FORCE = 70.0  # N, the hand's continuous grasp force; a move pushes with up to this
STALL_LAG = 0.003  # m behind the reference, at under half the commanded speed: the fingers are blocked
STALL_S = 0.1  # blocked this long: a grasp has its object, a move has ended short (still pushing)
DONE_TOL = 0.0015  # m: a move has arrived
MOVE_SETTLE_S = 0.5  # a move at its width (its reference there) that hasn't arrived by then has ended short of it


@dataclass(frozen=True)
class Command:
    kind: str  # "move" | "grasp"
    speed: float
    width: float = 0.0  # a move's
    force: float = 0.0  # a grasp's

    def as_dict(self) -> dict:
        if self.kind == "move":
            return {"kind": "move", "width_m": self.width, "speed_mps": self.speed}
        return {"kind": "grasp", "speed_mps": self.speed, "force_n": self.force}


def move(width: float, speed: float) -> Command:
    return Command("move", min(max(float(speed), 0.01), MAX_SPEED), width=min(max(float(width), 0.0), MAX_WIDTH))


def grasp(speed: float, force: float) -> Command:
    return Command("grasp", min(max(float(speed), 0.01), MAX_SPEED), force=min(max(float(force), 0.1), MAX_FORCE))


class FrankaHand:
    """Runs the commands; `step` gives the servo's (target width, closing force limit, squeezing, stiff) for one physics
    step: squeezing, the fingers close with that force whatever their width; stiff, a move's servo."""

    def __init__(self, width: float):
        self.reset(width)

    def reset(self, width: float) -> None:
        self.cmd = move(width, MAX_SPEED)
        self.sent = self.cmd  # the last command sent (what the robot knows it asked for)
        self.pending: Command | None = None
        self.busy = False
        self.ref = float(width)
        self.stalled_s = 0.0
        self.squeezing = False
        self.at_goal_s = 0.0

    def send(self, cmd: Command) -> None:
        self.sent = cmd
        if self.busy:
            self.pending = cmd
        else:
            self._start(cmd, None)

    def _start(self, cmd: Command, width: float | None) -> None:
        self.cmd, self.busy, self.stalled_s, self.squeezing, self.at_goal_s = cmd, True, 0.0, False, 0.0
        if width is not None:
            self.ref = width

    def step(self, h: float, width: float, speed: float) -> tuple[float, float, bool, bool]:
        if not self.busy and self.pending is not None:
            cmd, self.pending = self.pending, None
            self._start(cmd, width)
        cmd = self.cmd
        if self.busy:
            goal = cmd.width if cmd.kind == "move" else 0.0
            d = goal - self.ref
            self.ref += max(-cmd.speed * h, min(cmd.speed * h, d))
            behind = (self.ref - width) * (1.0 if goal >= width else -1.0)
            stopped = behind > STALL_LAG and abs(speed) < 0.5 * cmd.speed
            self.stalled_s = self.stalled_s + h if stopped else 0.0
            if cmd.kind == "move":
                self.at_goal_s = self.at_goal_s + h if self.ref == goal else 0.0
                if self.ref == goal and abs(width - goal) < DONE_TOL:
                    self.busy = False
                elif self.ref == goal and (self.stalled_s >= STALL_S or self.at_goal_s >= MOVE_SETTLE_S):
                    self.busy = False  # stopped short of it: done, and still pushing toward it
            elif self.stalled_s >= STALL_S:
                self.busy, self.squeezing = False, True  # grasped: the force stays on
            elif self.ref == 0.0 and width < DONE_TOL:
                self.busy = False  # closed on nothing
        force = cmd.force if cmd.kind == "grasp" else MAX_FORCE
        return self.ref, force, self.squeezing, cmd.kind == "move"
