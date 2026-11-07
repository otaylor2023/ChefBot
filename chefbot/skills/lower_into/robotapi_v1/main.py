"""lower_into (robot-api-v1): the agent's lower_into tool, agent/tools/manip.py HandheldTools.lower_into, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.handheld.lower_into(id, wait)
