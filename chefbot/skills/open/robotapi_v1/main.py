"""open (robot-api-v1): the agent's open tool, agent/tools/manip.py ManipTools.open, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.manip.open(id, wait)
