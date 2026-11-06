"""Recipe text -> Recipe via one structured OpenAI call (plus an optional repair call).

The key comes from OPENAI_API_KEY only. ReplayExtractor serves recorded responses (tests, --replay, --rescore).
Another provider is one more BaseExtractor subclass with a _call(system, user) -> LLMCall.
"""
from __future__ import annotations
from .config import DEFAULT_RECIPE_MODEL, DEFAULT_REASONING_EFFORT
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from .schema import Recipe, SourceDocument, VERB_TABLE

PROMPT_VERSION = "0.6"  # one fully stepped path; other options as quoted alternative rules


def _state_keys() -> str:
    keys = sorted({k for r in VERB_TABLE["verbs"].values() for k in r.get("sets", {})})
    return ", ".join(keys)


def _verb_lines() -> str:
    lines = []
    for name, rule in VERB_TABLE["verbs"].items():
        tags = []
        if "heat_source" in rule.get("needs", []):
            tags.append("needs a heated container as a `require` input")
        if rule.get("target") == "equipment":
            tags.append("equipment step: `require` the container, no output")
        lines.append(f"- {name}: {rule.get('desc', '')}" + (f" [{'; '.join(tags)}]" if tags else ""))
    lines.append("- other: nothing above fits; put the source verb in `verb_raw`")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You convert a cooking recipe into a structured, checkable representation for a kitchen robot planner.
A symbolic validator will execute your output step by step, so it must be internally consistent.

MODEL
- Ingredients are declared once, with snake_case ids. Containers (pan, pot, bowl, oven, tin) are declared once.
- Every step consumes inputs and produces ONE NEW output id (never reuse an id; e.g. onion -> onion_diced -> onion_cooked).
- An input is `consume` for food (with `fraction` of that item's original amount: 1.0 = all of it, 0.5 = half,
  "the remaining half" = 0.5) or `require` for equipment that must already be in some state (a hot pan, a preheated oven).
- Cook/bake/melt steps must `require` a container that an earlier heat/preheat step heated. Heating is checked
  automatically: do NOT add required_state for it.
- required_state may only use keys that verbs set: {_state_keys()}. Example: bake requires the oven with
  temp_c = the preheat temperature; dividing batter requires the tin with prepared = true.
- `container` on a step = where its output ends up.
- Steps are listed in the order the recipe gives them. Step ids are s1, s2, s3, ...
- Every ingredient, container and output id is unique: never give a container or an output the same id as an ingredient.

VERBS (closed set)
{_verb_lines()}

FAITHFULNESS RULES (most important)
1. evidence.quote must be an EXACT, contiguous, copy-pasted substring of the recipe text: same case, same punctuation,
   no ellipses, no paraphrase. Quote the shortest clause that supports the element (a few words, not the whole sentence).
2. prov="stated" when the text says it. prov="inferred" when you had to add it (e.g. a heat step implied by "saute");
   then quote may be null and `reason` is required.
3. NEVER invent numbers. If a temperature, time, heat level or amount is not given, leave the field null and describe the
   gap in `params.unresolved`, `fraction_raw`, or `until.kind="unresolved"` with `until.raw`.
   Vague amounts ("some of the sauce", "a splash") -> fraction null, fraction_raw = the words.
4. Preparation stated in the ingredient list ("2 eggs, beaten") -> ingredient `egg` plus a `beat` step whose evidence has
   implied_by="ingredient_list" and quotes the ingredient line.
5. Represent the recipe AS WRITTEN. Do not fix it. If it never cooks something that should be cooked, do not add a
   cooking step; the validator will report it. Omit pure serving/plating steps ("serve immediately").
6. needs_cooking = true only for ingredients unsafe to eat raw (raw egg, raw meat, poultry, fish).
7. Optional steps/ingredients ("optionally", "if desired") -> optional=true.
   ONE PATH: when the recipe offers options the cook picks between (several fillings, two methods, a vegetarian
   version), write steps only for the first-described option (or the one the recipe recommends). Never make the cook
   do every option in sequence. Record each other option as ONE rule kind="alternative" whose quote is the whole
   contiguous passage describing it (it may be long); do not write its steps.
   A short either/or where each side is one or two steps ("broil or bake") may instead be a choice: alternatives are
   lists of step ids, default = index of the first.
8. final_items = the id of the one dish the main path produces with optional steps skipped (not a garnished variant). List real ambiguities (not missing numbers) in `ambiguities`.
9. Declare every ingredient from the ingredient list, even if the instructions never use it.
10. Every distinct action of the main path is a step: do not merge separate actions or summarise a long recipe.
    Skip pure commentary (tips about storage, reheating, serving suggestions, equipment brands, the author's story).
    Short headings that only announce the next steps ("Cook the first layer.", "Make the sauce.") are not steps.
11. `rules` record what the text says that is not a step, each with an exact quote:
    kind="avoid" for constraints ("do not let the soup boil", "don't over-knead"),
    kind="contingency" for if/then instructions ("if the dough shrinks back, let it rest 10 minutes"),
    kind="alternative" for options the steps don't take (rule 7; "if using dashi powder, add it to boiling water"),
    kind="repeat" for repetition ("repeat with the remaining rice", "repeat until all the dough is used"); applies_to =
    the repeated steps.
    applies_to = ids of the affected steps (empty for the whole recipe). Never invent rules.
"""

REPAIR_PROMPT = """The validator ran your extraction and reported the findings below.

Fix ONLY extraction mistakes: wrong or missing ids, quotes that are not exact substrings, missing heat steps the text
implies, inputs consumed twice, steps you skipped that the text states, wrong verbs. If a finding says a cooking step is
inferred, DELETE that step: the recipe does not say to cook it, and the validator will report that gap to the user.

Do NOT change the recipe's meaning to make a finding disappear. Findings marked RECIPE describe the recipe itself
(for example an ingredient the recipe never cooks); leave them unless you missed a step the text explicitly states.
Any step you ADD must have prov="stated" with an exact quote; inferred additions are rejected.

Return the complete corrected extraction.

FINDINGS
{findings}

YOUR PREVIOUS EXTRACTION
{previous}
"""


def user_message(source: SourceDocument) -> str:
    return f"RECIPE TEXT\n<<<\n{source.text}\n>>>"


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0       # includes reasoning tokens (billed as output)
    latency_s: float = 0.0
    reasoning_tokens: int = 0
    cost_usd: float | None = None  # provider-reported cost (OpenRouter) or 0.0 (local); None = use pricing.yaml


@dataclass
class LLMCall:
    recipe: Recipe | None
    raw: str
    usage: Usage
    error: str | None = None

    def to_json(self) -> dict:
        return {"raw": self.raw, "usage": asdict(self.usage), "error": self.error}


class BaseExtractor:
    model: str = "?"
    # extract(source, retry=True) is called after unusable output; subclasses may change sampling for it.

    def _call(self, system: str, user: str) -> LLMCall:
        raise NotImplementedError

    def extract(self, source: SourceDocument, retry: bool = False, prev_error: str | None = None) -> LLMCall:
        return self._call(SYSTEM_PROMPT, user_message(source))

    def repair(self, source: SourceDocument, previous: Recipe, findings: str) -> LLMCall:
        msg = user_message(source) + "\n\n" + REPAIR_PROMPT.format(
            findings=findings, previous=previous.model_dump_json(indent=1))
        return self._call(SYSTEM_PROMPT, msg)

    @staticmethod
    def _parse(raw: str, usage: Usage) -> LLMCall:
        try:
            return LLMCall(Recipe.model_validate_json(raw), raw, usage)
        except Exception as e:  # schema-invalid output is a result to report, not a crash
            return LLMCall(None, raw, usage, f"schema_error: {e}")


class CappedExtractor(BaseExtractor):
    """Output cap that scales with the recipe, and retry behaviour after truncation or a degenerate loop.
    Subclasses set max_output_tokens / retry_temperature and read self.cap and self.retrying in _call."""
    max_output_tokens: int = 8000
    retry_temperature: float = 0.4
    retrying: bool = False
    cap: int = 8000

    def cap_for(self, source) -> int:
        return min(32000, max(self.max_output_tokens, int(4000 + 1.6 * len(source.text))))

    def extract(self, source, retry: bool = False, prev_error: str | None = None):
        self.cap = self.cap_for(source)
        truncated = bool(prev_error and prev_error.startswith("truncated"))
        loop = bool(prev_error and "loop" in prev_error)
        if retry and truncated and not loop:
            self.cap = min(32000, self.cap * 2)
        self.retrying = retry and not (truncated and not loop)
        try:
            return super().extract(source)
        finally:
            self.retrying = False

    def repair(self, source, previous, findings):
        self.cap = self.cap_for(source)
        return super().repair(source, previous, findings)


class OpenAIExtractor(CappedExtractor):
    """Responses API with a strict JSON schema; we parse the text ourselves so that usage and raw output are kept
    even when the output is unusable (a truncated or looping response is still billed).
    - temperature is dropped automatically if the model rejects it (reasoning models)
    - output cap scales with the recipe: max(max_output_tokens, 4000 + 1.6 * source chars), at most 32000
      (observed on 26 web recipes: ~1.5 output tokens per source character; 35-40 step recipes need 7-9k tokens)
    - retry after truncation at the cap: cap x2. Retry after a whitespace loop: `retry_temperature`
      (at temperature 0 a degenerate loop repeats at exactly the same character)"""

    def __init__(self, model: str = DEFAULT_RECIPE_MODEL, temperature: float | None = 0.0, reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
                 max_output_tokens: int = 8000, retry_temperature: float = 0.4):
        from openai import OpenAI
        from openai.lib._pydantic import to_strict_json_schema
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client, self.model, self.temperature = OpenAI(), model, temperature
        self.reasoning_effort, self.max_output_tokens, self.retry_temperature = reasoning_effort, max_output_tokens, retry_temperature
        self.text_format = {"format": {"type": "json_schema", "name": "Recipe", "strict": True,
                                       "schema": to_strict_json_schema(Recipe)}}
        self.retrying = False
        self.cap = max_output_tokens

    def _request(self, system, user, kwargs):
        return self.client.responses.create(model=self.model, instructions=system, input=user, text=self.text_format,
                                            max_output_tokens=self.cap, **kwargs)

    def _call(self, system, user):
        import openai
        t = time.perf_counter()
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.retry_temperature if self.retrying else self.temperature
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        try:
            resp = self._request(system, user, kwargs)
        except openai.BadRequestError as e:
            if "temperature" not in kwargs or "temperature" not in str(e):
                raise
            self.temperature = None
            kwargs.pop("temperature")
            resp = self._request(system, user, kwargs)
        u = resp.usage
        details = getattr(u, "output_tokens_details", None) if u else None
        usage = Usage(self.model, getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0,
                      time.perf_counter() - t, getattr(details, "reasoning_tokens", 0) or 0)
        raw = resp.output_text or ""
        if getattr(resp, "status", None) == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None) or "unknown"
            tail = raw[-200:]
            loop = " (degenerate whitespace loop)" if len(tail) == 200 and not tail.strip() else ""
            return LLMCall(None, raw, usage, f"truncated: {reason}{loop} after {usage.output_tokens} output tokens")
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "refusal":
                    return LLMCall(None, getattr(c, "refusal", ""), usage, "refusal")
        return self._parse(raw, usage)


class ReplayExtractor(BaseExtractor):
    """Serves recorded responses in order: [extraction, repair, ...]. For tests and offline demos."""

    def __init__(self, paths: list[str | Path]):
        self.paths, self.model, self.calls = [Path(p) for p in paths], "replay", 0

    def _call(self, system, user):
        path = self.paths[min(self.calls, len(self.paths) - 1)]
        self.calls += 1
        return self._parse(path.read_text(), Usage("replay:" + path.name))


def make_extractor(model: str, temperature: float | None = 0.0, reasoning_effort: str | None = None,
                   max_output_tokens: int = 8000) -> BaseExtractor:
    """model spec -> extractor.
      gpt-* / o1.. / openai:<m>        OpenAI Responses API          (OPENAI_API_KEY)
      openrouter:<vendor/model>       OpenRouter chat completions   (OPENROUTER_API_KEY)
      ollama:<name[:tag]>             local Ollama /api/chat        (OLLAMA_HOST, default http://localhost:11434)
    """
    from .providers import OllamaExtractor, OpenRouterExtractor
    if model.startswith("ollama:"):
        return OllamaExtractor(model.removeprefix("ollama:"), temperature, reasoning_effort, max_output_tokens)
    if model.startswith("openrouter:"):
        return OpenRouterExtractor(model.removeprefix("openrouter:"), temperature, reasoning_effort, max_output_tokens)
    if model.startswith(("claude", "anthropic:")):
        raise ValueError(f"{model!r}: no direct Anthropic extractor; use openrouter:anthropic/<model>")
    return OpenAIExtractor(model.removeprefix("openai:"), temperature, reasoning_effort, max_output_tokens)
