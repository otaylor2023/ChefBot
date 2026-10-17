"""Source adapters: anything -> SourceDocument. New formats (PDF, image) are new functions here; nothing downstream changes."""
from __future__ import annotations
import html
import json
import re
import urllib.request
from pathlib import Path
from .schema import SourceDocument, Section

_HEADINGS = {
    "ingredients": re.compile(r"^\s*#*\s*ingredients?\s*:?\s*$", re.I),
    "instructions": re.compile(r"^\s*#*\s*(instructions|directions|method|steps|preparation)\s*:?\s*$", re.I),
}


def _sections(text: str) -> list[Section]:
    marks: list[tuple[int, int, str]] = []  # (heading_start, body_start, kind)
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        for kind, rx in _HEADINGS.items():
            if rx.match(stripped):
                marks.append((pos, pos + len(line), kind))
        pos += len(line)
    if not marks:  # no headings: everything after a short title line counts as instructions
        first = text.split("\n", 1)[0]
        if len(first) < 80 and not first.rstrip().endswith((".", "!", "?")) and "\n" in text:
            return [Section(kind="other", start=0, end=len(first) + 1),
                    Section(kind="instructions", start=len(first) + 1, end=len(text))]
        return [Section(kind="instructions", start=0, end=len(text))]
    out = []
    if marks[0][0] > 0 and text[:marks[0][0]].strip():
        out.append(Section(kind="other", start=0, end=marks[0][0]))
    for i, (_, body, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append(Section(kind=kind, start=body, end=end))
    return out


def from_text(text: str) -> SourceDocument:
    text = text.replace("\r\n", "\n")
    return SourceDocument(text=text, sections=_sections(text))


def from_file(path: str | Path) -> SourceDocument:
    return from_text(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ URL / JSON-LD
def _find_recipe(node):
    if isinstance(node, list):
        for n in node:
            if (r := _find_recipe(n)) is not None:
                return r
    elif isinstance(node, dict):
        t = node.get("@type")
        if t == "Recipe" or (isinstance(t, list) and "Recipe" in t):
            return node
        for key in ("@graph", "mainEntity"):
            if key in node and (r := _find_recipe(node[key])) is not None:
                return r
    return None


def _instructions(node) -> list[str]:
    if isinstance(node, str):
        return [s for s in (x.strip() for x in re.split(r"\n+", _strip_tags(node))) if s]
    if isinstance(node, list):
        return [s for n in node for s in _instructions(n)]
    if isinstance(node, dict):
        if node.get("@type") == "HowToSection":
            return _instructions(node.get("itemListElement", []))
        return _instructions(node.get("text") or node.get("name") or "")
    return []


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).replace("\xa0", " ").strip()


def from_html(page: str) -> SourceDocument:
    for block in re.findall(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', page, re.S | re.I):
        try:
            recipe = _find_recipe(json.loads(block))
        except json.JSONDecodeError:
            continue
        if recipe:
            ingredients = [_strip_tags(i) for i in recipe.get("recipeIngredient", [])]
            steps = _instructions(recipe.get("recipeInstructions", []))
            text = (f"{_strip_tags(recipe.get('name', ''))}\n\nIngredients\n"
                    + "".join(f"- {i}\n" for i in ingredients)
                    + "\nInstructions\n" + "".join(f"{n}. {s}\n" for n, s in enumerate(steps, 1)))
            hints = {k: recipe[k] for k in ("recipeYield", "prepTime", "cookTime", "totalTime") if k in recipe}
            doc = from_text(text)
            doc.structured_hints = {"source": "json-ld", **hints}
            return doc
    # No JSON-LD: fall back to visible text. Coverage will be noisy (ads, comments), so this is flagged.
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    doc = from_text(re.sub(r"\n\s*\n+", "\n\n", _strip_tags(re.sub(r"<(br|p|li|h\d)[^>]*>", "\n", body))))
    doc.structured_hints = {"source": "html-text-fallback"}
    return doc


def from_url(url: str, timeout: float = 20) -> SourceDocument:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ChefBot recipe-ir research prototype)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        page = r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")
    doc = from_html(page)
    doc.structured_hints["url"] = url
    return doc


def load_source(spec: str) -> SourceDocument:
    if spec.startswith(("http://", "https://")):
        return from_url(spec)
    p = Path(spec)
    if p.suffix.lower() in (".html", ".htm"):
        return from_html(p.read_text(encoding="utf-8", errors="replace"))
    return from_file(p)
