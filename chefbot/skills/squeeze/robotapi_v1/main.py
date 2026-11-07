"""squeeze (robot-api-v1): the agent's squeeze tool, agent/tools/manip.py HandheldTools.squeeze, on the session's RobotAPI."""


def run(ctx, wait: bool = True):
    return ctx.handheld.squeeze(wait)
