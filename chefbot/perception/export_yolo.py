"""Recorded frames + their detections -> Ultralytics YOLO segmentation datasets, for training a detector on our own
kitchen later. One folder per pass (images/ + labels/ side by side, the layout Ultralytics reads); a split is just a
list of folders, so making a new train/val split is choosing folders, and adding data is dropping in another pass.

    python -m chefbot.perception.export_yolo pass DETS/p2_layout3/pole STREAM/p2_layout3 --out DATA/p2_layout3 \\
        [--labels labels.yaml] [--min-score 0.3]
    python -m chefbot.perception.export_yolo split --root DATA --train p1_layout2 p2_layout3 --val p3_moved_unseen

Labels are the detector's (pseudo-labels): correct them (a labeling tool on labels/*.txt) before trusting a model
trained on them. labels.yaml: {rename: {detector label: ours}, ignore: [...], keep_only: [...]} as for memory_map.
Images are hard-linked when on the same disk (no copy), else copied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def _our_label(label: str, labels: dict) -> str | None:
    label = labels.get("rename", {}).get(label, label)
    if label in set(labels.get("ignore", [])):
        return None
    keep = labels.get("keep_only")
    return label if not keep or label in keep else None


def export_pass(det_dir, pass_dir, out, labels=None, min_score: float = 0.3, names: list[str] | None = None,
                width: int = 1280, height: int = 800, cam: str = "pole") -> list[str]:  # fmt: skip
    """Write out/images/*.png and out/labels/*.txt (class x y x y ... normalized polygon per object). Returns the
    class names (given `names` first, new ones appended)."""
    labels, names = labels or {}, list(names or [])
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for line in (Path(det_dir) / "detections.jsonl").read_text().splitlines():
        d = json.loads(line)
        rows = []
        for det in d["dets"]:
            lab = _our_label(det["label"], labels)
            if lab is None or det["score"] < min_score or len(det.get("poly") or []) < 3:
                continue
            if lab not in names:
                names.append(lab)
            xy = " ".join(f"{x / width:.5f} {y / height:.5f}" for x, y in det["poly"])
            rows.append(f"{names.index(lab)} {xy}")
        src = Path(pass_dir) / cam / "color" / d["file"]
        dst = out / "images" / d["file"]
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        (out / "labels" / d["file"].replace(".png", ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
    (out / "names.json").write_text(json.dumps(names))
    return names


def write_split(root, train: list[str], val: list[str]) -> Path:
    root = Path(root)
    names: list[str] = []
    for f in train + val:
        for n in json.loads((root / f / "names.json").read_text()):
            if n not in names:
                names.append(n)
    for f in train + val:  # one class list across folders: renumber each folder's labels to it
        local = json.loads((root / f / "names.json").read_text())
        if local != names[: len(local)]:
            remap = {i: names.index(n) for i, n in enumerate(local)}
            for t in (root / f / "labels").glob("*.txt"):
                lines = [l.split(" ", 1) for l in t.read_text().splitlines() if l.strip()]
                t.write_text("".join(f"{remap[int(c)]} {rest}\n" for c, rest in lines))
            (root / f / "names.json").write_text(json.dumps(names))
    data = {"path": str(root.resolve()), "train": [f"{f}/images" for f in train], "val": [f"{f}/images" for f in val],
            "names": dict(enumerate(names))}  # fmt: skip
    p = root / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("pass")
    e.add_argument("det_dir", help="detect_frames.py's OUT/<pass>/<cam>")
    e.add_argument("pass_dir", help="the recorded stream/<pass>")
    e.add_argument("--out", required=True)
    e.add_argument("--labels")
    e.add_argument("--min-score", type=float, default=0.3)
    s = sub.add_parser("split")
    s.add_argument("--root", required=True)
    s.add_argument("--train", nargs="+", required=True)
    s.add_argument("--val", nargs="+", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "pass":
        labels = yaml.safe_load(Path(a.labels).read_text()) if a.labels else None
        meta = json.loads((Path(a.pass_dir) / "pole" / "meta.json").read_text())["color"]
        names = export_pass(a.det_dir, a.pass_dir, a.out, labels, a.min_score, width=meta["width"], height=meta["height"])
        print(f"{a.out}: {len(list(Path(a.out, 'images').glob('*.png')))} images, classes {names}")
    else:
        print(write_split(a.root, a.train, a.val).read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
