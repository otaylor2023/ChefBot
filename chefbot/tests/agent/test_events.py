"""Timers, monitors, safety monitors and the event-aware wait (AGENT_SPEC §6, §7.4)."""

import math

from chefbot.agent.check.checker import CheckResult
from chefbot.agent.events.bus import EventBus
from chefbot.agent.plan.model import Plan
from chefbot.agent.plan.store import PlanStore
from chefbot.agent.tools.time_tools import TimeTools
from chefbot.interface.clock import ManualClock
from chefbot.interface.types import MotionStatus, Pose2D
from chefbot.tests.agent.fakes import FakeRobot

PLAN = Plan.from_dict({"goal": "egg", "steps": [{"id": "s1", "goal": "Cook", "done_when": "egg: set"}]})


class TimedRobot(FakeRobot):
    """A FakeRobot whose motions take `motion_s` of run time; wait() lets time pass."""

    def __init__(self, clock: ManualClock, motion_s: float = 10.0):
        super().__init__(Pose2D(0.0, 0.0, 0.0))
        self.clock, self.motion_s = clock, motion_s
        self.hold = True
        self._started = None

    def _start(self, goal, effect=None, data=None):
        super()._start(goal, effect, data)
        self._started = self.clock.now()

    def wait(self, timeout_s=None):
        if self._pending is not None:
            step = self.motion_s if timeout_s is None else timeout_s
            self.clock.sleep(min(step, self._started + self.motion_s - self.clock.now()))
            if self.clock.now() >= self._started + self.motion_s - 1e-9:
                self.finish()
        return self.get_status()


def setup(check=None, motion_s=10.0):
    clock = ManualClock()
    robot = TimedRobot(clock, motion_s)
    bus = EventBus(clock, check)
    store = PlanStore(PLAN, clock.now)
    return clock, robot, bus, TimeTools(robot, bus, clock, store)


def test_timers_fire_once_at_their_time():
    clock, robot, bus, tools = setup()
    assert tools.set_timer("egg", 150, "s1").success
    clock.sleep(149.0)
    assert bus.poll() == []
    clock.sleep(1.0)
    [event] = bus.poll()
    assert event.kind == "timer" and event.step_id == "s1" and event.time == 150.0
    assert bus.poll() == [] and not bus.active()


def test_cancelled_and_restarted_timers():
    clock, robot, bus, tools = setup()
    tools.set_timer("egg", 60)
    assert tools.cancel_timer("egg").success
    assert tools.cancel_timer("egg").error == "unknown_timer"
    tools.set_timer("egg", 60)
    clock.sleep(50)
    tools.set_timer("egg", 60)  # restart
    clock.sleep(20)
    assert bus.poll() == []


def test_monitor_checks_at_its_interval_and_stops_on_pass():
    world = {"text": "egg: raw"}
    calls = []

    def check(step_id, condition):
        calls.append(clock.now())
        return CheckResult(condition in world["text"], world["text"])

    clock, robot, bus, tools = setup(check)
    assert tools.watch("s1", every_s=20, timeout_s=300).success
    for _ in range(3):
        clock.sleep(20)
        assert bus.poll() == []
    world["text"] = "egg: set"
    clock.sleep(20)
    [event] = bus.poll()
    assert event.kind == "monitor" and event.data["passed"] and event.step_id == "s1"
    assert calls == [20.0, 40.0, 60.0, 80.0]
    assert not bus.monitors  # stopped; the step is not completed for the agent


def test_monitor_times_out_with_the_last_reason():
    clock, robot, bus, tools = setup(lambda step, cond: CheckResult(False, "egg: raw"))
    tools.watch("s1", every_s=30, timeout_s=60)
    events = []
    while not events and clock.now() < 200:
        clock.sleep(min(10, bus.next_due() - clock.now()))
        events = bus.poll()
    [event] = events
    assert event.data["timeout"] and event.data["reason"] == "egg: raw" and event.time == 60.0


def test_safety_monitors_post_events():
    clock, robot, bus, tools = setup()
    bus.add_safety(lambda now: [("burner_1 on with nothing on it", {"burner": "burner_1"})] if now > 30 else [])
    assert bus.poll() == []
    clock.sleep(31)
    [event] = bus.poll()
    assert event.kind == "safety" and event.data == {"burner": "burner_1"}


def test_wait_with_nothing_to_wait_for():
    clock, robot, bus, tools = setup()
    result = tools.wait()
    assert not result.success and result.error == "nothing_to_wait_for"


def test_wait_returns_on_timeout_letting_time_pass():
    clock, robot, bus, tools = setup()
    result = tools.wait(45)
    assert result.success and result.data["woke_for"] == "timeout" and clock.now() == 45.0


def test_wait_returns_on_the_first_event_and_fast_forwards_to_it():
    clock, robot, bus, tools = setup()
    tools.set_timer("egg", 150)
    tools.set_timer("toast", 90)
    result = tools.wait()
    assert result.data["woke_for"] == "event" and [e["data"]["timer"] for e in result.data["events"]] == ["toast"]
    assert math.isclose(clock.now(), 90.0)
    assert tools.wait(10).data["woke_for"] == "timeout"


def test_wait_returns_when_the_motion_ends():
    clock, robot, bus, tools = setup(motion_s=12.0)
    robot.go_to_pose(Pose2D(1.0, 0.0, 0.0))
    result = tools.wait()
    assert result.data["woke_for"] == "motion_done" and result.data["status"] == MotionStatus.SUCCESS
    assert math.isclose(clock.now(), 12.0, abs_tol=0.1)


def test_an_event_wakes_a_wait_on_a_motion_which_keeps_going():
    clock, robot, bus, tools = setup(motion_s=100.0)
    tools.set_timer("egg", 30)
    robot.go_to_pose(Pose2D(1.0, 0.0, 0.0))
    result = tools.wait()
    assert result.data["woke_for"] == "event" and result.data["status"] == MotionStatus.MOVING
    assert 30.0 <= clock.now() < 31.0
    assert robot.get_status().status == MotionStatus.MOVING


def test_a_motion_left_running_becomes_an_event_when_it_ends():
    clock, robot, bus, tools = setup(motion_s=5.0)
    robot.go_to_pose(Pose2D(1.0, 0.0, 0.0))
    tools.note_motion('go_to("stove")', True)
    tools.poll_motion()
    assert not bus.has_pending()
    robot.wait()  # time passes, the motion ends
    tools.poll_motion()
    [event] = bus.drain()
    assert event.kind == "motion_done" and 'go_to("stove") ended: success' in event.message
