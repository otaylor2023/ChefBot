import json

from chefbot.perception.live_map import Follow


def test_follow_reads_only_whole_new_lines(tmp_path):
    cam = tmp_path / "pole"
    cam.mkdir()
    (cam / "meta.json").write_text("{}")
    (tmp_path / "state.jsonl").write_text("")
    fl = cam / "frames.jsonl"
    fl.write_text(json.dumps({"n": 1}) + "\n" + '{"n": ')  # the recorder is mid-line
    f = Follow(tmp_path)
    assert [x["n"] for x in f.new_frames()] == [1]
    assert f.new_frames() == []
    with open(fl, "a") as fo:
        fo.write('2}\n{"n": 3}\n')
    assert [x["n"] for x in f.new_frames()] == [2, 3]
    assert not f.ended()
    (tmp_path / "summary.json").write_text("{}")
    assert f.ended()
