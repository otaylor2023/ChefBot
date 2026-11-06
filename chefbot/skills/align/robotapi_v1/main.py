"""align (robot-api-v1): the agent's align tool, agent/tools/nav.py NavTools.align, on the session's RobotAPI."""


def run(ctx, id: str, wait: bool = True):
    return ctx.nav.align(id, wait)
