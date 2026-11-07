"""release (robot-api-v1): the agent's release tool, agent/tools/manip.py HandheldTools.release, on the session's RobotAPI."""


def run(ctx, wait: bool = True):
    return ctx.handheld.release(wait)
