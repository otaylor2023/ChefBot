"""pour (robot-api-v1): the agent's pour tool, agent/tools/manip.py HandheldTools.pour, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.handheld.pour(id, wait)
