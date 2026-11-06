"""turn_on (robot-api-v1): the agent's turn_on tool, agent/tools/manip.py KitchenTools.turn_on, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.kitchen.turn_on(id, wait)
