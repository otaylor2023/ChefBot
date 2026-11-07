"""A session: one agent in one env with one robot API. It decides what the agent is offered: the skills in its
registry (agent.yaml `skills:`, each pinned to one registered version) that could run in this env and speak this API,
and the tools it lists (`tools:`) that work with this API.

    session = Session.open(agents.load("kitchen_planner"), env="symbolic")   # its interface: the agent's first
    session.skills      # [Skill]: go_to/robotapi_v1, align/robotapi_v1, ...
    session.tools       # ("detect", "nearby", ..., "watch")

Refused: an interface the agent or its harness does not run with, an env no skill knows, a skill named like one of
its tools (the LLM calls each by its name).
"""

from __future__ import annotations

from dataclasses import dataclass

from chefbot import skills as skillbook
from chefbot import tools as toolbox
from chefbot.agents import Agent


@dataclass(frozen=True)
class Session:
    agent: Agent
    env: str
    interface: str

    @classmethod
    def open(cls, agent: Agent, env: str, interface: str | None = None) -> "Session":
        session = cls(agent, env, interface or agent.interfaces[0])
        session.validate()
        return session

    @property
    def skills(self) -> list[skillbook.Skill]:
        return [s for s in self.agent.skills if self.env in s.envs and s.interface == self.interface]

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(t for t in self.agent.tools if self.interface in toolbox.get(t).interfaces)

    def describe(self) -> dict:
        """For the run log."""
        return {"env": self.env, "interface": self.interface, "skills": [s.path for s in self.skills], "tools": list(self.tools)}

    def validate(self) -> None:
        a = self.agent
        if self.interface not in a.interfaces:
            raise ValueError(f"agent {a.name} runs with {', '.join(a.interfaces)}, not {self.interface}")
        harness_interfaces = getattr(a.harness, "INTERFACES", ())
        if self.interface not in harness_interfaces:
            raise ValueError(f"harness {a.harness_name} runs {', '.join(harness_interfaces)} sessions, not {self.interface}"
                             f" ({a.harness.__name__}.INTERFACES)")  # fmt: skip
        if self.env not in skillbook.ENVS:
            raise ValueError(f"no env {self.env!r}; envs: {', '.join(skillbook.ENVS)}")
        clash = sorted({s.name for s in self.skills} & set(self.tools))
        if clash:
            raise ValueError(f"agent {a.name}: skills and tools both named {', '.join(clash)}: rename the skill in its skills:")
