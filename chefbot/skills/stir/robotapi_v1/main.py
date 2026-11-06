"""stir (robot-api-v1): the agent's stir tool, agent/tools/manip.py HandheldTools.stir, on the session's RobotAPI."""


def run(ctx, seconds: float, wait: bool = True):
    return ctx.handheld.stir(seconds, wait)
