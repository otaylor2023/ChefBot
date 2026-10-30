"""Reference sets: photos and specs of real products, split dev / val / test. They live outside the repo (CHEFBOT_REFS,
default ~/chefbot_refs), one folder per family and split:

    <refs>/<family>/<split>/<id>/spec.json     {"name", "url", "length_m", "mass_kg", "tips", "material", "lock", ...}
    <refs>/<family>/<split>/<id>/photo*.jpg    one or more photos
    <refs>/<family>/heldout/rubric.md          the val/test judge's rubric (not the researcher's)

dev is the researcher's (copied into the workspace); val and test are hidden from it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

TILE = (300, 300)


def root() -> Path:
    return Path(os.environ.get("CHEFBOT_REFS", Path.home() / "chefbot_refs"))


def split_dir(family: str, split: str) -> Path:
    return root() / family / split


def items(family: str, split: str) -> list[dict]:
    out = []
    d = split_dir(family, split)
    for item in sorted(p for p in d.glob("*") if p.is_dir()) if d.exists() else []:
        spec = json.loads((item / "spec.json").read_text()) if (item / "spec.json").exists() else {}
        photos = sorted(p for p in item.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if photos:
            out.append({"id": item.name, "dir": str(item), "photos": [str(p) for p in photos], **spec})
    return out


def montage(family: str, split: str, out: Path, max_items: int = 20) -> Path | None:
    """The split's first photo of each item, captioned with its name and length, in one image."""
    its = items(family, split)[:max_items]
    if not its:
        return None
    cols = min(5, len(its))
    rows = math.ceil(len(its) / cols)
    sheet = Image.new("RGB", (cols * TILE[0], rows * (TILE[1] + 24)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, it in enumerate(its):
        try:
            im = ImageOps.pad(Image.open(it["photos"][0]).convert("RGB"), TILE, color=(255, 255, 255))
        except OSError:
            continue
        x, y = (k % cols) * TILE[0], (k // cols) * (TILE[1] + 24)
        sheet.paste(im, (x, y + 24))
        length = f", {it['length_m'] * 100:.0f} cm" if isinstance(it.get("length_m"), (int, float)) else ""
        dr.text((x + 4, y + 5), f"{it.get('name', it['id'])[:40]}{length}", fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out


def spec_table(family: str, split: str) -> str:
    """The split's specs as a markdown table (for the researcher's dev set)."""
    its = items(family, split)
    keys = ["name", "length_m", "mass_kg", "tips", "material", "lock", "url"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for it in its:
        lines.append("| " + " | ".join(str(it.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)
