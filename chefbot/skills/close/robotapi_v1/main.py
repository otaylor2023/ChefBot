"""close (robot-api-v1): the agent's close tool, agent/tools/manip.py ManipTools.close, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.manip.close(id, wait)
