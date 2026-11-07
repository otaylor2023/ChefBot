import json

import yaml

from chefbot.perception.export_yolo import export_pass, write_split

POLY = [[10, 10], [30, 10], [30, 30]]


def _pass(tmp, name, dets):
    det_dir, pass_dir = tmp / f"dets_{name}", tmp / f"stream_{name}"
    det_dir.mkdir()
    (pass_dir / "pole" / "color").mkdir(parents=True)
    (pass_dir / "pole" / "color" / "000000.png").write_bytes(b"png")
    (det_dir / "detections.jsonl").write_text(json.dumps({"n": 0, "file": "000000.png", "dets": dets}) + "\n")
    return det_dir, pass_dir


def test_export_and_split_share_one_class_list(tmp_path):
    a = _pass(tmp_path, "a", [{"label": "cup", "score": 0.9, "poly": POLY}, {"label": "mug", "score": 0.9, "poly": POLY},
                              {"label": "cup", "score": 0.1, "poly": POLY}])  # fmt: skip
    b = _pass(tmp_path, "b", [{"label": "tomato", "score": 0.5, "poly": POLY}])
    root = tmp_path / "data"
    assert export_pass(*a, root / "a", labels={"rename": {"mug": "cup"}}, width=100, height=100) == ["cup"]
    assert (root / "a" / "labels" / "000000.txt").read_text().splitlines()[0].startswith("0 0.10000 0.10000")
    export_pass(*b, root / "b", width=100, height=100)
    assert (root / "b" / "labels" / "000000.txt").read_text().startswith("0 ")  # tomato is class 0 in b alone
    data = yaml.safe_load(write_split(root, ["a"], ["b"]).read_text())
    assert data["names"] == {0: "cup", 1: "tomato"} and data["val"] == ["b/images"]
    assert (root / "b" / "labels" / "000000.txt").read_text().startswith("1 ")  # renumbered to the shared list
