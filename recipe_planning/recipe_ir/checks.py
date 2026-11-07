"""Deterministic checks outside the state engine: evidence grounding, faithfulness, rules, uncited text,
dependency edges."""
from __future__ import annotations
import re
from .schema import Recipe, SourceDocument
from .engine import Violation


def check_evidence(recipe: Recipe, source: SourceDocument) -> list[Violation]:
    """Find each quote in the source (sets evidence.span in place). Repeated phrases resolve in reading order.
    A capitalisation-only difference is accepted with a warning ('skillet' vs a title's 'Skillet')."""
    text, low, out, cursor = source.text, source.text.lower(), [], 0
    for oid, ev in recipe.evidenced():
        if ev.prov == "inferred" and not (ev.reason or ev.implied_by):
            out.append(Violation("inference_without_reason", oid, "prov=inferred needs reason or implied_by"))
        if ev.quote is None:
            if ev.prov == "stated":
                out.append(Violation("stated_without_evidence", oid, "prov=stated needs a quote"))
            continue
        hits = [m.start() for m in re.finditer(f"(?={re.escape(ev.quote)})", text)]       # overlapping matches
        if not hits:
            hits = [m.start() for m in re.finditer(f"(?={re.escape(ev.quote.lower())})", low)]
            if not hits:
                out.append(Violation("quote_not_found", oid, repr(ev.quote)))
                continue
            out.append(Violation("quote_case_mismatch", oid,
                                 f"{ev.quote!r} matches {text[hits[0]:hits[0] + len(ev.quote)]!r} only ignoring case"))
        start = min(hits, key=lambda h: (h < cursor, abs(h - cursor)))
        ev._span = (start, start + len(ev.quote))
        cursor = ev._span[1]
    return out


def check_faithfulness(recipe: Recipe, rules: dict) -> list[Violation]:
    """An inferred cook/bake step turns a real gap ('the egg is never cooked') into a silently valid plan.
    Allowed: implied_by=ingredient_list ('2 cups cooked rice'). Inferred heat/preheat steps are fine."""
    return [Violation("inferred_cooking_step", s.id,
                      f"{s.verb} of {[i.ref for i in s.inputs if i.mode == 'consume']} is inferred, not stated; "
                      "represent the recipe as written and let the validator report the gap")
            for s in recipe.steps
            if rules.get(s.verb, {}).get("cooks") and s.evidence.prov == "inferred" and not s.evidence.implied_by]


_COOKED = re.compile(
    r"\b(cook\w*|fr(?:y|ies|ied|ying)|stir[- ]?fr\w*|saut[eé]\w*|sear\w*|brown\w*|simmer\w*|boil\w*|bak(?:e|es|ed|ing)|"
    r"roast\w*|broil\w*|grill\w*|toast\w*|poach\w*|steam\w*|scrambl\w*|melt\w*|carameli[sz]\w*|reduc\w*|bubbl\w*|"
    r"sizzl\w*|char\w*|blister\w*|wilt\w*|warm\w*|heat\w*|crisp\w*|render\w*|thicken\w*|soften\w*|set(?! aside)|sets|"
    r"done|golden|tender|translucent|fragrant|microwave\w*|blanch\w*|brais\w*|stew\w*|smok\w*|puff\w*|opaque)\b", re.I)
_TIMED = re.compile(r"\b(\d+|a|an|one|two|few|couple)\b[^.\n]{0,12}\b(sec|second|min|minute|hour|hr)s?\b", re.I)


def _sentence(text: str, span) -> str:
    """The sentence (or list line) that contains a quote's span."""
    a, b = span
    start = max(text.rfind(". ", 0, a), text.rfind("\n", 0, a)) + 1
    m = re.search(r"[.!?](?:\s|$)|\n", text[b:])
    return text[start: b + (m.end() if m else len(text) - b)]


def check_cooking_support(recipe: Recipe, source: SourceDocument, rules: dict) -> list[Violation]:
    """A stated cook/bake step must quote a sentence that says the food is cooked: a cooking word ("scramble",
    "simmer", "until golden", "until set") or a time ("3 minutes"). "Immediately add the eggs" is a transfer; calling
    it cooking turns a real gap into a valid plan (the planted-gap failure). Run after check_evidence (needs spans)."""
    out = []
    for s in recipe.steps:
        if not rules.get(s.verb, {}).get("cooks") or s.evidence.prov != "stated" or s.evidence.span is None:
            continue
        sent = _sentence(source.text, s.evidence.span)
        if not (_COOKED.search(sent) or _TIMED.search(sent)):
            out.append(Violation("unsupported_cooking_step", s.id,
                                 f"{s.verb} quotes {s.evidence.quote!r}, but that sentence never says the food is "
                                 "cooked (no cooking word, doneness cue or time)"))
    return out


def check_rules(recipe: Recipe) -> list[Violation]:
    """Rules only record what the text says: stated, and pointing at real steps."""
    steps = {s.id for s in recipe.steps}
    out = []
    for r in recipe.rules:
        bad = [sid for sid in r.applies_to if sid not in steps]
        if bad:
            out.append(Violation("unknown_rule_target", r.id, f"rule applies_to unknown step id(s) {bad}"))
        if r.evidence.prov != "stated":
            out.append(Violation("inferred_rule", r.id, "rules must quote the recipe; drop rules the text doesn't state"))
    return out


_DUR = re.compile(r"(\d+(?:\.\d+)?)(?:\s*(?:to|-|–|or)\s*(\d+(?:\.\d+)?))?\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.I)
_TEMP = re.compile(r"(\d{2,3})\s*(?:°|º|degrees?)?\s*([CF])\b", re.I)


def _window(text: str, span, pad: int = 250) -> str:
    return text if span is None else text[max(0, span[0] - pad): span[1] + pad]


_NUMWORDS = {"a": "1", "an": "1", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
             "seven": "7", "eight": "8", "nine": "9", "ten": "10", "fifteen": "15", "twenty": "20", "thirty": "30",
             "a couple of": "2", "a couple": "2", "couple of": "2"}
_NUMWORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _NUMWORDS), key=len, reverse=True))
                         + r")\s+(?=(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)\b)", re.I)


def _numbers_as_digits(text: str) -> str:
    """'a minute' -> '1 minute', 'two to three minutes' -> '2 to 3 minutes', 'half an hour' -> '30 minutes'.
    'A few minutes' stays words: a vague amount must stay unresolved, not become a number."""
    text = re.sub(r"\bhalf an hour\b", "30 minutes", text, flags=re.I)
    text = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?=(?:to|or|-|–)\s+\w+\s+(?:seconds?|minutes?|hours?)\b)",
                  lambda m: _NUMWORDS[m.group(1).lower()] + " ", text, flags=re.I)
    return _NUMWORD_RE.sub(lambda m: _NUMWORDS[m.group(1).lower()] + " ", text)


def _duration_ok(seconds: float, text: str) -> bool:
    text = _numbers_as_digits(text)
    for m in _DUR.finditer(text):
        word = m.group(3).lower()
        unit = 1 if word.startswith("s") else (60 if word.startswith("m") else 3600)
        lo = float(m.group(1)) * unit
        hi = float(m.group(2)) * unit if m.group(2) else lo
        if lo * 0.9 <= seconds <= hi * 1.1:
            return True
    hours = [float(m.group(1)) * 3600 for m in _DUR.finditer(text) if m.group(3).lower().startswith("h")]
    mins = [float(m.group(1)) * 60 for m in _DUR.finditer(text) if m.group(3).lower().startswith("m")]
    return any(abs(h + m - seconds) <= 60 for h in hours for m in mins)   # "1 hour 30 minutes"


def _temp_ok(temp_c: float, text: str) -> bool:
    for m in _TEMP.finditer(text):
        n = float(m.group(1))
        c = n if m.group(2).upper() == "C" else (n - 32) * 5 / 9
        if abs(c - temp_c) <= 3:
            return True
    return False


def check_params(recipe: Recipe, source: SourceDocument) -> list[Violation]:
    """Rule 3 of the prompt, enforced: a heat level, temperature or duration must be written in the source near the
    step (the step's quote +-250 chars; the whole text for a step without a quote). A heat level may also come from
    the heat step that heated a container this step requires. Run after check_evidence."""
    text, out = source.text, []
    heated_by: dict[str, str] = {}          # container -> text window of the heat step that last heated it
    has = lambda value, win: re.search(r"\b" + re.escape(value.strip()).replace(r"\-", "[- ]?") + r"\b", win, re.I)
    for st in recipe.steps:
        win = _window(text, st.evidence.span)
        p = st.params
        if st.verb in ("heat", "preheat"):
            heated_by.update({i.ref: win for i in st.inputs if i.mode == "require"})
        # a heat level on a cook step may come from the step that heated its pan ("over high heat" 3 sentences up)
        pan_wins = [heated_by[i.ref] for i in st.inputs if i.mode == "require" and i.ref in heated_by]
        if p.heat and not has(p.heat, win) and not any(has(p.heat, w) for w in pan_wins):
            out.append(Violation("unstated_param", st.id, f"heat={p.heat!r} is not in the source near this step; "
                                 "leave it null and say what is missing in params.unresolved"))
        if p.temp_c is not None and not _temp_ok(p.temp_c, win):
            out.append(Violation("unstated_param", st.id, f"temp_c={p.temp_c} is not in the source near this step"))
        if p.duration_s is not None and not _duration_ok(p.duration_s, win):
            out.append(Violation("unstated_param", st.id, f"duration_s={p.duration_s} is not in the source near this step"))
    return out


# ---------------------------------------------------------------- coverage and uncited text
_FILLER = {"and", "then", "the", "a", "an", "or", "with", "to", "it", "in", "of", "until", "into"}


def instruction_coverage(recipe: Recipe, source: SourceDocument) -> tuple[float, list[tuple[int, int]]]:
    """Share of instruction-section text cited by some quote (or captured verbatim in until.raw /
    fraction_raw), plus the uncited gaps. Gaps of only connectives ('and', 'then') are dropped."""
    text = source.text
    spans = [ev.span for _, ev in recipe.evidenced() if ev.span]
    for raw in [s.until.raw for s in recipe.steps if s.until and s.until.raw] + \
               [i.fraction_raw for s in recipe.steps for i in s.inputs if i.fraction is None and i.fraction_raw]:
        if (i := text.find(raw)) != -1:
            spans.append((i, i + len(raw)))
    spans.sort()
    total = covered = 0
    gaps: list[tuple[int, int]] = []
    for sec in (s for s in source.sections if s.kind == "instructions"):
        pos = sec.start
        for a, b in spans:
            a, b = max(a, sec.start), min(b, sec.end)
            if b <= pos:
                continue
            if a > pos:
                gaps.append((pos, a))
            covered += b - max(a, pos)
            pos = b
        if pos < sec.end:
            gaps.append((pos, sec.end))
        total += sec.end - sec.start
    words = lambda g: re.findall(r"[^\W_]+", g.lower())
    gaps = [(a, b) for a, b in gaps if any(w not in _FILLER for w in words(text[a:b]))]
    return (covered / total if total else 0.0), gaps


_ACTION_WORDS = set("""add put place wet sprinkle stir cut slice dice chop mince bring boil simmer cook bake heat preheat
mix combine whisk beat fold pour transfer drain rinse wash soak season cover remove turn flip roll knead shape form wrap
spread brush grease line let rest cool serve garnish top dissolve strain melt toast fry saute sauté grill roast blend puree
mash peel grate measure divide press fill repeat set return reduce adjust taste crack dip scoop squeeze sift cream punch
proof deflate score slash sear lower increase uncover discard reserve tear arrange insert shake skim stretch fluff spoon
drizzle toss pat thaw refrigerate chill freeze wipe""".split())
_LEAD = re.compile(r"^(\d+\.\s*)?((first|then|next|now|finally|meanwhile|after that|once [^,]+|while [^,]+|gently|"
                   r"carefully|slowly|quickly),?\s+)*", re.I)
_RULE = re.compile(r"^(do not|don't|dont|never|avoid|be careful|make sure|be sure|if|when)\b", re.I)
_COMMENT = re.compile(r"^(tip|note|optional|alternatively|here,|i |i'|my |you can store|store |this |these |it's best|"
                      r"it‘s best|serve|enjoy|refrigerate|reheat)\b", re.I)


def classify_uncited(text: str) -> list[tuple[str, str]]:
    """Label uncited sentences action / rule / heading / commentary (English keyword heuristic).
    heading: a short sentence opening a numbered step ("3. Cook the first layer."); the steps after it carry it."""
    out = []
    after_number = False
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        if re.fullmatch(r"\s*\d+\.\s*", sent):
            after_number = True
            continue
        opens_step = after_number or bool(re.match(r"^\s*\d+\.\s+\S", sent))
        after_number = False
        s = _LEAD.sub("", sent.strip()).strip(" -•")
        if len(s) < 4:
            continue
        if opens_step and len(s.split()) <= 5 and s.endswith("."):
            kind = "heading"
        elif _COMMENT.match(s):
            kind = "commentary"
        elif _RULE.match(s):
            kind = "rule"
        elif re.split(r"[\s,]+", s.lower(), maxsplit=1)[0] in _ACTION_WORDS:
            kind = "action"
        else:
            kind = "commentary"
        out.append((kind, sent.strip()))
    return out


# ---------------------------------------------------------------- graph
def dependency_edges(recipe: Recipe, rules: dict) -> set[tuple[str, str]]:
    """(before, after) step pairs: dataflow (uses an earlier output), equipment state (requires a container an
    earlier equipment step changed, e.g. preheat -> bake), and write-after-read (an equipment step that changes a
    container waits for the earlier steps that required it, e.g. cook -> turn_off). Unconnected steps may run in
    parallel."""
    producer: dict[str, str] = {}
    last_writer: dict[str, str] = {}
    readers: dict[str, set[str]] = {}       # steps that used a container since its last state change
    edges: set[tuple[str, str]] = set()
    containers = {c.id for c in recipe.containers}
    for step in recipe.steps:
        for inp in step.inputs:
            if inp.ref in producer:
                edges.add((producer[inp.ref], step.id))
            if inp.ref in last_writer and last_writer[inp.ref] != step.id:
                edges.add((last_writer[inp.ref], step.id))
        if rules.get(step.verb, {}).get("target") == "equipment":
            for ref in (i.ref for i in step.inputs if i.ref in containers):
                # write after read: "remove from heat" must wait for the cooking that needed the heat
                edges.update((r, step.id) for r in readers.pop(ref, set()) if r != step.id)
                last_writer[ref] = step.id
        else:
            for i in step.inputs:
                if i.mode == "require" and i.ref in containers:
                    readers.setdefault(i.ref, set()).add(step.id)
        if step.output:
            producer[step.output] = step.id
    return edges
