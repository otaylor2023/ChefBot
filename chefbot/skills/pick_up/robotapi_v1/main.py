"""pick_up (robot-api-v1): the agent's pick tool, agent/tools/manip.py ManipTools.pick, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.manip.pick(id, wait)
