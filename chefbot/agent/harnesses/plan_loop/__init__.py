"""The plan loop (AGENT_SPEC §8): one tool call per turn against a plan, each step's done_when checked."""

from chefbot.agent.harnesses.plan_loop.runner import AgentRunner, AgentSession

# The robot APIs it runs sessions with. Not chefbot-v1 yet: a chefbot-v1 skill is an episode (runtime/episode.py) on a
# host that resets the scene at its start, and nothing yet keeps one scene across the skills of a run.
INTERFACES = ("robot-api-v1",)

__all__ = ["AgentRunner", "AgentSession", "INTERFACES"]
