"""The Franka Hand's command logic (hand.py) on fingers that follow the servo exactly, or stop on an object."""

from chefbot.embodiments.twin import hand as H

DT = 0.002


def run(hand, width, seconds, block=None):
    """Fingers that go where the servo says, unless an object `block` m wide is between them. Returns the width."""
    for _ in range(int(seconds / DT)):
        ref, _, squeezing, _ = hand.step(DT, width, 0.0 if (block is not None and width <= block) else 1.0)
        new = block if squeezing else ref
        if block is not None:
            new = max(new, block)
        width = new
    return width


def test_move_runs_at_its_speed():
    hand = H.FrankaHand(0.08)
    hand.send(H.move(0.02, 0.05))
    assert abs(run(hand, 0.08, 0.6) - 0.05) < 1e-3
    assert abs(run(hand, 0.05, 1.0) - 0.02) < 1e-6 and not hand.busy


def test_commands_wait_and_the_latest_wins():
    hand = H.FrankaHand(0.08)
    hand.send(H.move(0.04, 0.1))
    hand.send(H.move(0.06, 0.1))
    hand.send(H.move(0.07, 0.1))
    w = run(hand, 0.08, 0.3)
    assert abs(w - 0.05) < 2e-3  # still on the first
    assert abs(run(hand, w, 1.0) - 0.07) < 1e-6
    assert hand.sent.width == 0.07


def test_grasp_squeezes_with_its_force_and_a_blocked_move_keeps_pushing():
    hand = H.FrankaHand(0.08)
    hand.send(H.grasp(0.05, 12.0))
    run(hand, 0.08, 1.5, block=0.05)
    assert hand.squeezing and not hand.busy
    assert hand.step(DT, 0.05, 0.0)[1] == 12.0
    hand.send(H.move(0.0, 0.05))
    w = run(hand, 0.05, 2.0, block=0.05)
    ref, force, squeezing, stiff = hand.step(DT, 0.05, 0.0)
    assert not hand.busy and abs(w - 0.05) < 1e-6  # it ended where the object stopped it ...
    assert ref == 0.0 and force == H.MAX_FORCE and stiff and not squeezing  # ... still pushing toward 0, up to 70 N


def test_limits():
    assert H.grasp(1.0, 500).force == H.MAX_FORCE and H.grasp(1.0, 500).speed == H.MAX_SPEED
    assert H.move(0.5, 0.0).width == H.MAX_WIDTH and H.move(-1, 0.0).speed == 0.01
