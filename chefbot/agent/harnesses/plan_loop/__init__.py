"""The plan loop (AGENT_SPEC §8): one tool call per turn against a plan, each step's done_when checked."""

from chefbot.agent.harnesses.plan_loop.runner import AgentRunner, AgentSession

# The robot APIs it runs sessions with: robot-api-v1 skills run in its process (ToolRegistry); a chefbot-v1 skill is
# one sandboxed episode per call on a host that stays open for the run, continuing its scene (runtime/agent_skills.py).
INTERFACES = ("robot-api-v1", "chefbot-v1")

__all__ = ["AgentRunner", "AgentSession", "INTERFACES"]
