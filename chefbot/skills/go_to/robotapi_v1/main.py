"""go_to (robot-api-v1): the agent's go_to tool, agent/tools/nav.py NavTools.go_to, on the session's RobotAPI."""


def run(ctx, place: str, wait: bool = True):
    return ctx.nav.go_to(place, wait)
