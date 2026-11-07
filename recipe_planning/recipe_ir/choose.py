"""Pick the recipe: from what is in the kitchen, the user's constraints, and what the robot can do.

The proactive step before extraction (FINDINGS.md, next step 8). Deterministic v0 over the corpus text, no LLM:
- hard constraints are CHECKS, not preferences: a candidate that breaks one (diet, an excluded ingredient, too many
  missing items, an action the robot can't do when `robot="strict"`) is never returned;
- missing ingredients are reported, never bought (the reward-hacking lesson: "dinner, fast" must not become
  "order takeout");
- the robot screen reads action cues in the text (cut, bake, season, cover, ...) against the same swappable skills
  file as actions.feasibility. It is a pre-screen: the exact per-step report comes after extraction.
"""
from __future__ import annotations
import re
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .adapters import from_text

STAPLES = {"salt", "pepper", "black pepper", "water", "ice", "oil", "olive oil", "vegetable oil", "cooking spray"}
_STAPLE_RE = re.compile(r"^(?:(?:kosher|sea|table|fine|grain|coarse|flaky|cooking|ground|freshly ground)\s+)*(?:salt|black pepper|"
                        r"white pepper|pepper|salt pepper|salt black pepper|water|ice|cooking spray|"
                        r"(?:olive|vegetable|canola|neutral|avocado|virgin olive|extra virgin olive)?\s*oil)$")
# compounds whose head noun alone would match the wrong pantry item ("butter" must not cover "peanut butter")
DISTINCT = ("peanut butter", "almond butter", "apple butter", "almond milk", "coconut milk", "oat milk", "soy milk",
            "coconut cream", "sweet potato", "bell pepper", "chili pepper", "cream cheese", "rice vinegar", "rice paper",
            "rice noodle", "egg noodle", "tomato paste", "tomato sauce", "maple sugar", "brown sugar", "powdered sugar",
            "confectioner sugar", "coconut sugar", "wheat flour", "almond flour", "coconut flour", "rice flour",
            "semolina flour", "buckwheat flour", "chickpea flour", "bread flour", "cake flour", "corn tortilla",
            "flour tortilla", "sour cream", "ice cream", "coconut oil", "sesame oil", "chili oil", "garlic powder",
            "onion powder", "chili powder", "baking powder", "baking soda", "soy sauce", "fish sauce", "hot sauce")

_QTY = re.compile(r"^[\s\-•*]*(?:about|approx\.?|approximately)?\s*[\d¼½¾⅓⅔⅛⅜⅝⅞/.,\s\-–]*(?:to\s+[\d¼½¾⅓⅔/.]+)?\s*", re.I)
_UNITS = r"(?:cups?|c\.|tablespoons?|tbsps?|tbs|tsps?|teaspoons?|grams?|g|kg|ml|l|liters?|litres?|ounces?|oz|pounds?|lbs?|lb|" \
         r"cloves?|cans?|jars?|packages?|pkgs?|sticks?|slices?|pinch(?:es)?|dash(?:es)?|bunch(?:es)?|handfuls?|sprigs?|" \
         r"heads?|stalks?|pieces?|inch(?:es)?|quarts?|pints?|sheets?|large|medium|small|whole|heaping|scant|level)\b\.?"
_PREP = r"(?:fresh(?:ly)?|finely|thinly|roughly|coarsely|lightly|freshly|chopped|diced|minced|sliced|julienned|grated|" \
        r"shredded|crushed|ground|toasted|dried|frozen|canned|cooked|uncooked|raw|peeled|seeded|halved|quartered|cubed|" \
        r"softened|melted|beaten|room[- ]temperature|cold|warm|hot|packed|divided|plus more|extra[- ]virgin|unsalted|" \
        r"salted|boneless|skinless|organic|large|medium|small|ripe|optional|to taste|for serving|for garnish)\b"
_OPTIONAL = re.compile(r"\b(optional|for garnish|to garnish|for serving|to serve|if desired|if using)\b", re.I)

MEAT = r"\b(wagyu|kobe|mutton|venison|goat|ribs?|brisket|sirloin|tenderloin|pastrami|chicken|beef|pork|bacon|sausage|ham|steak|turkey|lamb|veal|duck|prosciutto|pancetta|chorizo|salami|pepperoni|" \
       r"meat|gelatin|anchov\w*|fish|salmon|tuna|cod|tilapia|halibut|shrimp|prawns?|scallops?|crab|lobster|clams?|" \
       r"mussels?|oyster sauce|fish sauce)\b"
ANIMAL = MEAT + r"|\b(eggs?|milk|butter|cheese|cream|yogurt|yoghurt|honey|ghee|mayonnaise|mayo|buttermilk|parmesan|feta|mozzarella)\b"

# action cue in the instructions -> recipe verb (the vocabulary the skills file uses)
CUES = {
    "cut": r"\b(chop|dice|slice|mince|julienne|cube|cut|halve|quarter)\w*",
    "bake": r"\b(bake[sd]?|baking|roast(ed|ing)?|broil\w*|oven)\b",
    "season": r"\b(season(ed|ing)? with|sprinkle\w*|salt and pepper to taste)\b",
    "cover": r"\b(cover(ed)? (the|with)|lid)\b",
    "blend": r"\b(blend\w*|food processor|puree\w*)\b",
    "knead": r"\b(knead\w*|dough)\b",
    "drain": r"\b(drain\w*|strain\w*)\b",
    "cook": r"\b(cook|simmer|saut[eé]\w*|fry|fried|frying|boil\w*|scrambl\w*|stir[- ]fry\w*|sear\w*)\b",
    "heat": r"\b(heat|preheat the (pan|skillet|wok))\b",
    "mix": r"\b(stir|mix|whisk|toss|combine|fold)\w*",
    "crack": r"\bcrack\w*",
    "melt": r"\bmelt\w*",
    "transfer": r"\b(add|pour|transfer|place)\b",
}


def _singular(w: str) -> str:
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("oes", "ches", "shes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalize_name(raw: str) -> str:
    """'- 3 cups julienned broccoli stems' -> 'broccoli stem'; '2 large eggs, beaten' -> 'egg'."""
    s = re.sub(r"\([^)]*\)", " ", raw)
    s = s.split(",")[0]
    s = re.split(r"\b(?:for|plus|or)\b", s)[0] if not s.lower().startswith(("for", "or")) else s
    s = _QTY.sub("", s)
    for _ in range(3):
        s = re.sub(r"^\s*(?:" + _UNITS + r")\s*(?:of\s+)?", "", s, flags=re.I)
        s = re.sub(r"\b" + _PREP, " ", s, flags=re.I)
    words = [_singular(w) for w in re.findall(r"[a-zà-ÿ]+", s.lower()) if w not in ("and", "a", "an", "the", "of", "into", "in")]
    return " ".join(words)


@lru_cache(maxsize=4096)
def _parse_cached(text: str) -> tuple:
    return tuple(_parse(text))


def parse_ingredients(text: str) -> list[dict[str, Any]]:
    return [dict(x) for x in _parse_cached(text)]


def _parse(text: str):
    doc = from_text(text)
    sec = next((text[s.start:s.end] for s in doc.sections if s.kind == "ingredients"), "")
    out = []
    for line in sec.splitlines():
        if not line.strip().startswith(("-", "•", "*")):
            continue
        name = normalize_name(line)
        if name and not re.match(r"^(see|note|few|a few|some|fillings?|toppings?)\b", name):
            out.append({"raw": line.strip("-•* ").strip(), "name": name, "optional": bool(_OPTIONAL.search(line))})
    return out


SYNONYMS = {"green onion": "scallion", "spring onion": "scallion", "coriander": "cilantro", "garbanzo": "chickpea",
            "garbanzo bean": "chickpea", "cornflour": "cornstarch", "courgette": "zucchini", "aubergine": "eggplant",
            "capsicum": "bell pepper", "prawn": "shrimp", "rocket": "arugula", "caster sugar": "sugar",
            "granulated sugar": "sugar", "all purpose flour": "flour", "plain flour": "flour", "plain all purpose flour": "flour",
            "heavy whipping cream": "heavy cream", "double cream": "heavy cream", "minced beef": "ground beef"}


_SYN_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SYNONYMS), key=len, reverse=True)) + r")\b")


@lru_cache(maxsize=65536)
def canon(name: str) -> str:
    return _SYN_RE.sub(lambda m: SYNONYMS[m.group(1)], name)


@lru_cache(maxsize=65536)
def _pantry_name(p: str) -> str:
    return canon(" ".join(_singular(w) for w in re.findall(r"[a-z]+", p.lower())))


def have(pantry: list[str], name: str) -> str | None:
    """The pantry item that covers this ingredient: all its words appear in the ingredient's name
    ('egg' covers 'egg'; 'onion' covers 'red onion'; 'oil' covers 'olive oil')."""
    name = canon(name)
    toks = name.split()
    while len(toks) > 1 and toks[-1] in PARTS:    # 'lemon juice', 'garlic clove': the part names the pantry item
        toks = toks[:-1]
    compound = next((c for c in DISTINCT if c in name), None)
    for p in pantry:
        pn = _pantry_name(p)
        if not pn or not set(pn.split()) <= set(toks) or toks[-1] not in pn.split():
            continue                     # every pantry word in the name, and the name's head noun among them
        if compound and compound not in pn:
            continue
        return p
    return None


PARTS = {"juice", "zest", "clove", "leaf", "leave", "stem", "floret", "wedge", "slice", "sprig", "half", "piece", "chunk",
         "peel", "rind", "white", "yolk", "kernel", "segment"}


def is_staple(name: str, staples: set[str]) -> bool:
    return name in staples or bool(_STAPLE_RE.match(name))


@lru_cache(maxsize=4096)
def action_cues(text: str) -> frozenset[str]:
    doc = from_text(text)
    instr = next((text[s.start:s.end] for s in doc.sections if s.kind == "instructions"), text)
    return frozenset(verb for verb, pat in CUES.items() if re.search(pat, instr, re.I))


def robot_screen(cues: set[str], skills: dict) -> tuple[float, list[str]]:
    """Share of the recipe's action kinds the robot supports (partial counts half), and the blockers."""
    verbs = skills.get("verbs", {})
    if not cues:
        return 1.0, []
    score, blockers = 0.0, []
    for c in sorted(cues):
        st = verbs.get(c, {}).get("status", "unsupported")
        score += {"supported": 1.0, "partial": 0.5}.get(st, 0.0)
        if st == "unsupported":
            blockers.append(f"{c}: {verbs.get(c, {}).get('note', 'no skill')}")
    return round(score / len(cues), 3), blockers


@dataclass
class Prefs:
    diet: str | None = None                   # vegetarian | vegan
    exclude: list[str] = field(default_factory=list)
    max_missing: int = 2
    max_ingredients: int | None = None
    robot: str = "rank"                       # rank (prefer doable) | strict (only recipes with no blocker) | ignore
    staples: set[str] = field(default_factory=lambda: set(STAPLES))


def constraint_violations(text: str, ings: list[dict], prefs: Prefs) -> list[str]:
    title = text.strip().splitlines()[0] if text.strip() else ""
    names = " | ".join([title.lower()] + [i["raw"].lower() for i in ings if not i["optional"]])   # raw: "(or beef)"
    bad = []
    if prefs.diet in ("vegetarian", "vegan") and (m := re.search(MEAT, names)):
        bad.append(f"{prefs.diet}: {m.group(0)}")
    if prefs.diet == "vegan" and (m := re.search(ANIMAL, names)):
        bad.append(f"vegan: {m.group(0)}")
    for x in prefs.exclude:
        if re.search(r"\b" + re.escape(_singular(x.lower())) + r"s?\b", names):
            bad.append(f"excluded: {x}")
    if prefs.max_ingredients and len(ings) > prefs.max_ingredients:
        bad.append(f"{len(ings)} ingredients > {prefs.max_ingredients}")
    return bad


def assess(name: str, text: str, pantry: list[str], prefs: Prefs, skills: dict) -> dict[str, Any]:
    ings = parse_ingredients(text)
    needed = [i for i in ings if not i["optional"] and not is_staple(i["name"], prefs.staples)]
    missing = [i["raw"] for i in needed if not have(pantry, i["name"])]
    used = sorted({p for i in needed if (p := have(pantry, i["name"]))})
    fit, blockers = robot_screen(action_cues(text), skills)
    bad = constraint_violations(text, ings, prefs)
    if not ings:
        bad.append("no ingredient list to check against the pantry")
    if needed and not used:
        bad.append("uses nothing in the kitchen")
    if len(missing) > prefs.max_missing:
        bad.append(f"{len(missing)} missing > {prefs.max_missing}")
    if prefs.robot == "strict" and blockers:
        bad.append("robot can't: " + "; ".join(blockers))
    return {"name": name, "ok": not bad, "violations": bad, "missing": missing, "pantry_used": used,
            "n_needed": len(needed), "robot_fit": fit, "robot_blockers": blockers}


def score(a: dict, prefs: Prefs) -> float:
    """Use what the kitchen has, need little else, and fit the robot. Each missing item costs more than a used one
    earns, so a recipe that needs a shopping trip ranks below one you can cook now."""
    robot = 0.0 if prefs.robot == "ignore" else a["robot_fit"]
    return len(a["pantry_used"]) - 1.5 * len(a["missing"]) + robot


def rank_key(a: dict, prefs: Prefs) -> tuple:
    return (-score(a, prefs), len(a["missing"]), a["n_needed"], a["name"])


def choose(recipes: dict[str, str], pantry: list[str], prefs: Prefs, skills: dict, k: int = 5) -> list[dict[str, Any]]:
    """recipes: name -> recipe text. Returns the top k candidates that pass every hard constraint, best first."""
    ok = [a for n, t in recipes.items() if (a := assess(n, t, pantry, prefs, skills))["ok"]]
    return sorted(ok, key=lambda a: rank_key(a, prefs))[:k]


def load_corpus(corpus_dir: Path, names: list[str] | None = None) -> dict[str, str]:
    paths = [corpus_dir / f"{n}.txt" for n in names] if names else sorted(corpus_dir.glob("*.txt"))
    return {p.stem: p.read_text(encoding="utf-8") for p in paths if p.exists()}
