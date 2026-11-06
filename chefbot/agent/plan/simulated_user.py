"""A simulated user for the planner's questions (evaluation; HW1's personas): an LLM told what the user asked and
what they mean (the task file's `user:`), answering one question at a time in a sentence. It never plans."""

from __future__ import annotations

PROMPT = """You are a person who asked a kitchen robot for something. You said: "{request}".
What you meant (never say this list as a whole; answer only what you are asked): {knows}
The robot's planner asks you a question. Answer it in one short sentence, as that person would. If the question is
about something you have no opinion on, say so ("I don't mind")."""


class SimulatedUser:
    def __init__(self, llm, request: str, knows: str):
        self.llm, self.prompt = llm, PROMPT.format(request=request, knows=knows.strip())
        self.asked: list[tuple[str, str]] = []
        self.cost_usd = 0.0

    def __call__(self, question: str) -> str:
        r = self.llm.chat([{"role": "system", "content": self.prompt}, {"role": "user", "content": question}])
        self.cost_usd += r.cost_usd or 0.0
        answer = (r.text or "").strip() or "I don't mind."
        self.asked.append((question, answer))
        return answer
