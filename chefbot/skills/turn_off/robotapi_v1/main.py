"""turn_off (robot-api-v1): the agent's turn_off tool, agent/tools/manip.py KitchenTools.turn_off, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.kitchen.turn_off(id, wait)
