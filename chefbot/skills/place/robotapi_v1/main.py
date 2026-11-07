"""place (robot-api-v1): the agent's place tool, agent/tools/manip.py ManipTools.place, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.manip.place(id, wait)
