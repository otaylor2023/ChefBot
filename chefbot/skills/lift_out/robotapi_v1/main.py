"""lift_out (robot-api-v1): the agent's lift_out tool, agent/tools/manip.py HandheldTools.lift_out, on the session's RobotAPI."""


def run(ctx, wait: bool = True):
    return ctx.handheld.lift_out(wait)
