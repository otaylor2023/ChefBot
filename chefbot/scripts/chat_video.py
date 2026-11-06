"""A run's video with its conversation beside it (runs/<run>/chat_video.mp4).

    python -m chefbot.scripts.chat_video runs/<run>  [--min-turn 2.5] [--speed 2]

Left: the run's own video (record_rollout: overview, shoulder camera, map, caption), with the plan as the executor
saw it that turn. Right: the conversation as it happened, newest at the bottom. It opens on what each LLM is told
(the planner's and the executor's system prompts, the executor's tools, the judge's prompt, scrolled through), then
the user's request, the planner's user message, its reply and plan; then every turn of the loop at the moment the
video reaches it: the TASK STATE sent, the executor's words and tool call, the judge's verdict, the robot's result.

Everything comes from the run's files (planner.json, requests/, run.jsonl, checks/, video.mp4). A turn shorter than
--min-turn seconds is held on its last frame so its messages can be read; the clock in the corner is the run's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 20
LEFT_W, CHAT_X = 1120, 1136
BG, PANEL, LINE, INK, MUTED = (21, 25, 26), (29, 34, 36), (52, 60, 62), (228, 234, 230), (150, 162, 157)
COLORS = {"user": (185, 150, 230), "planner": (238, 125, 69), "exec": (108, 176, 238), "judge": (224, 123, 189),
          "robot": (92, 196, 137), "harness": (163, 173, 168)}  # fmt: skip
NAMES = {"user": "User", "planner": "Planner · Sonnet 5.5", "exec": "Executor · Sonnet 5.5", "judge": "Judge · Haiku 5.5",
         "robot": "Robot · simulator", "harness": "Harness · code"}  # fmt: skip
HARNESS_TOOLS = {"complete", "set_status", "add_step", "skip_step", "request_replan", "finish", "set_timer", "cancel_timer", "watch"}


def _font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size, index=index)
    except OSError:
        return ImageFont.load_default(size)


SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
F_BODY, F_BOLD, F_SMALL = _font(SANS, 20), _font(SANS, 17, 1), _font(SANS, 15)
F_MONO, F_MONO_S, F_TITLE = _font(MONO, 16), _font(MONO, 12), _font(SANS, 30, 1)


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(x.get("text", "") for x in content or [])


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    out = []
    for para in str(text).split("\n"):
        if not para:
            out.append("")
            continue
        avg = max(1, int(font.getlength("abcdefghij") / 10))
        lines = textwrap.wrap(para, max(10, width // avg), replace_whitespace=False, drop_whitespace=False) or [""]
        out += [l.rstrip() for l in lines]
    return out


@dataclass
class Msg:
    frm: str
    to: str | None
    body: str  # plain text
    code: str = ""  # mono text under it (a tool call, a verdict)
    dim: bool = False


def _msg_height(m: Msg, width: int) -> int:
    h = 34
    if m.body:
        h += len(_wrap(m.body, F_SMALL if m.dim else F_BODY, width)) * (20 if m.dim else 26)
    if m.code:
        h += len(_wrap(m.code, F_MONO, width - 16)) * 21 + 12
    return h + 12


def _draw_msg(d: ImageDraw.ImageDraw, m: Msg, x: int, y: int, width: int) -> None:
    c = COLORS[m.frm]
    h = _msg_height(m, width - 28)
    d.rounded_rectangle((x, y, x + width, y + h - 8), 8, fill=PANEL, outline=LINE)
    d.rectangle((x, y + 2, x + 5, y + h - 10), fill=c)
    head = NAMES[m.frm] + (f"  →  {NAMES[m.to]}" if m.to else "")
    d.text((x + 16, y + 8), NAMES[m.frm], font=F_BOLD, fill=c)
    if m.to:
        w0 = d.textlength(NAMES[m.frm], font=F_BOLD)
        d.text((x + 16 + w0 + 8, y + 9), "→", font=F_MONO, fill=MUTED)
        d.text((x + 16 + w0 + 34, y + 8), NAMES[m.to], font=F_BOLD, fill=COLORS[m.to])
    yy = y + 34
    if m.body:
        f, lh = (F_SMALL, 20) if m.dim else (F_BODY, 26)
        for line in _wrap(m.body, f, width - 28):
            d.text((x + 16, yy), line, font=f, fill=MUTED if m.dim else INK)
            yy += lh
    if m.code:
        lines = _wrap(m.code, F_MONO, width - 44)
        d.rounded_rectangle((x + 14, yy + 2, x + width - 12, yy + 8 + 21 * len(lines)), 5, fill=(38, 45, 47))
        for line in lines:
            d.text((x + 22, yy + 5), line, font=F_MONO, fill=INK)
            yy += 21
    del head


class Chat:
    """The right panel: messages stacked bottom up, newest at the bottom."""

    def __init__(self):
        self.msgs: list[Msg] = []
        self._cache: tuple[int, Image.Image] | None = None

    def image(self, title: str) -> Image.Image:
        if self._cache and self._cache[0] == len(self.msgs):
            return self._cache[1]
        width = W - CHAT_X - 16
        img = Image.new("RGB", (W - CHAT_X, H), BG)
        d = ImageDraw.Draw(img)
        y = H - 12
        for m in reversed(self.msgs):
            h = _msg_height(m, width - 28)
            y -= h
            if y + h < 56:
                break
            _draw_msg(d, m, 0, y, width)  # one cut off at the top scrolls out under the title
        d.rectangle((0, 0, W - CHAT_X, 50), fill=BG)
        d.text((0, 12), title, font=F_BOLD, fill=MUTED)
        d.line((0, 48, W - CHAT_X - 16, 48), fill=LINE)
        self._cache = (len(self.msgs), img)
        return img


def _scroll_image(title: str, color: tuple, text: str, width: int) -> Image.Image:
    lines = _wrap(text, F_MONO_S, width - 30)
    img = Image.new("RGB", (width, 60 + 18 * len(lines) + 40), PANEL)
    d = ImageDraw.Draw(img)
    d.text((14, 14), title, font=F_BOLD, fill=color)
    for i, line in enumerate(lines):
        d.text((14, 50 + 18 * i), line, font=F_MONO_S, fill=INK)
    return img


def _plan_block(state: str) -> str:
    if "PLAN" not in state:
        return ""
    block = state.split("PLAN", 1)[1].split("\n\nNOW", 1)[0].strip("\n")
    return "\n".join(l for l in block.splitlines() if not l.strip().startswith(("log:", "done so far:", "tried:", "note:")))


def _left(frame: np.ndarray | None, plan: str, clock: str, label: str) -> Image.Image:
    img = Image.new("RGB", (LEFT_W, H), BG)
    if frame is not None:
        f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.paste(f.crop((0, 0, 640, 480)).resize((560, 420)), (0, 0))
        img.paste(f.crop((640, 0, 1280, 480)).resize((560, 420)), (560, 0))
        img.paste(f.crop((1280, 0, 1760, 480)).resize((420, 420)), (0, 420))
        img.paste(f.crop((0, 480, 1760, 576)).resize((1120, 61)), (0, 846))
    d = ImageDraw.Draw(img)
    d.rectangle((424, 424, LEFT_W - 4, 840), fill=PANEL)
    d.text((440, 434), "THE PLAN, AS THE EXECUTOR SEES IT", font=F_BOLD, fill=MUTED)
    y = 466
    for line in plan.splitlines()[:24]:
        st = line.strip()
        col = COLORS["robot"] if " done " in line else COLORS["planner"] if " active " in line else INK if st[:1] == "s" else MUTED
        for part in _wrap(line, F_MONO_S, 650)[:2]:
            d.text((440, y), part, font=F_MONO_S, fill=col)
            y += 17
        if y > 826:
            break
    d.text((16, 924), label, font=F_TITLE, fill=INK)
    d.text((16, 972), clock, font=F_SMALL, fill=MUTED)
    return img


def render(run: Path, min_turn: float = 2.5, speed: int = 2, out: Path | None = None) -> Path:
    run = Path(run)
    rows = [json.loads(l) for l in (run / "run.jsonl").read_text().splitlines()]
    acts = [r for r in rows if r["kind"] == "action"]
    planner = json.loads((run / "planner.json").read_text()) if (run / "planner.json").exists() else None
    start = json.loads((run / "start.json").read_text())
    end = json.loads((run / "end.json").read_text())
    reqs = sorted((run / "requests").glob("*.json"))
    first = json.loads(reqs[0].read_text())
    cap = cv2.VideoCapture(str(run / "video.mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    n_src = len(frames)

    # where each turn starts in the source video (record_rollout: 2 s title hold; per call its motion, then a 1 s hold,
    # 2 s for wait / complete; waits don't render), scaled to the frames actually recorded
    hold = lambda s: s * src_fps / speed  # noqa: E731
    pos, starts = hold(2.0), []
    for a in acts:
        starts.append(pos)
        pos += (0 if a["tool"] == "wait" else a["duration_s"] * src_fps / speed) + hold(2.0 if a["tool"] in ("wait", "complete") else 1.0)
    scale = n_src / (pos + hold(3.0))
    starts = [int(s * scale) for s in starts] + [n_src]

    checks = {r["turn"]: json.loads((run / r["file"]).read_text()) for r in rows if r["kind"] == "check" and r.get("file")}
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out or run / "chat_video.mp4")],
                            stdin=subprocess.PIPE)  # fmt: skip
    chat = Chat()
    task = start.get("task", "run")
    label = f"ChefBot · {task.replace('_', ' ')} · {start.get('session', {}).get('env', '')}"

    def emit(left: Image.Image, right: Image.Image, n: int = 1) -> None:
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(left, (0, 0))
        canvas.paste(right, (CHAT_X, 0))
        raw = canvas.tobytes()
        for _ in range(n):
            proc.stdin.write(raw)

    # 1. the setup: what each LLM is told, scrolled through
    first_frame = frames[0] if frames else None
    intro_left = _left(first_frame, _plan_block(_text(first["messages"][-1]["content"])) if planner is None else "(the planner hasn't planned yet)",
                       "before the run", label)  # fmt: skip
    tools = "\n".join(f"{t['function']['name']}({', '.join(t['function']['parameters'].get('properties', {}))}): "
                      f"{t['function']['description']}" for t in first.get("tools") or [])  # fmt: skip
    from chefbot.agent.check.checker import JUDGE_PROMPT

    width = W - CHAT_X - 16
    pmsgs = (planner["calls"][0].get("messages") or []) if planner else []
    setup = []
    if pmsgs:
        setup.append(("Planner · system prompt (planner.md): the task, how cooking works, what a good plan is", COLORS["planner"], _text(pmsgs[0]["content"]), 11))
    setup += [("Executor · system prompt: how to act, find things, use the plan, stay safe", COLORS["exec"], _text(first["messages"][0]["content"]), 13),
              (f"Executor · its {len(first.get('tools') or [])} tools, as described to it", COLORS["exec"], tools, 9),
              ("Judge · system prompt: how a step is checked", COLORS["judge"], JUDGE_PROMPT, 4)]  # fmt: skip
    for title, color, text, secs in setup:
        img = _scroll_image(title, color, text, width)
        n = int(secs * FPS)
        travel = max(0, img.height - (H - 70))
        for i in range(n):
            y = int(travel * max(0.0, (i - FPS) / max(1, n - 2 * FPS))) if travel else 0
            right = Image.new("RGB", (W - CHAT_X, H), BG)
            dd = ImageDraw.Draw(right)
            dd.text((0, 12), "WHAT EACH LLM IS TOLD (the same every turn)", font=F_BOLD, fill=MUTED)
            right.paste(img.crop((0, y, width, min(img.height, y + H - 70))), (0, 60))
            emit(intro_left, right)

    # 2. the request and the plan
    title = "THE CONVERSATION"
    if planner:
        chat.msgs.append(Msg("user", "planner", f"“{planner['request']}”"))
        emit(intro_left, chat.image(title), int(2.5 * FPS))
        if len(pmsgs) > 1:
            um = _text(pmsgs[1]["content"])
            kitchen = um.split("ROBOT", 1)[0].replace("REQUEST: " + planner["request"], "").strip()
            skills = ", ".join(l[2:].split("(")[0] for l in um.split("ROBOT", 1)[1].strip().splitlines() if l.startswith("- "))
            places = [l.rstrip() for l in kitchen.splitlines()[2:] if l.strip()][:9]
            chat.msgs.append(Msg("harness", "planner", "the request, plus what the robot knows (KITCHEN) and what it can do (ROBOT):",
                                 "KITCHEN\n" + "\n".join(places) + f"\nROBOT  {skills}", dim=True))  # fmt: skip
            emit(intro_left, chat.image(title), int(4 * FPS))
        answers = dict((q, a) for q, a in planner["calls"][0].get("questions") or [])
        for t in planner["calls"][0]["transcript"]:
            a = t.get("args", {})
            if t["tool"] == "ask_user":
                chat.msgs.append(Msg("planner", "user", t.get("said") or "", f"ask_user(\"{a.get('question')}\")"))
                chat.msgs.append(Msg("user", "planner", f"“{answers.get(a.get('question'), '')}”"))
            elif t["tool"] == "submit_plan":
                plan = "\n".join(f"{s['id']}  {s['goal']}\n    done_when: {s['done_when']}" for s in a.get("steps", []))
                chat.msgs.append(Msg("planner", "harness", t.get("said") or "", f"submit_plan(goal=\"{a.get('goal')}\")\n{plan}"))
            else:
                chat.msgs.append(Msg("planner", "harness", t.get("said") or "", f"{t['tool']}({json.dumps(a)})"))
            emit(intro_left, chat.image(title), int(7 * FPS))
            if t.get("reply"):
                chat.msgs.append(Msg("harness", "planner", "", f"check_plan: {t['reply']}"))
                emit(intro_left, chat.image(title), int(1.5 * FPS))

    # 3. the loop, at the video's pace
    for i, a in enumerate(acts):
        req = json.loads((run / a["request"]).read_text()) if a.get("request") else None
        reply = (req or {}).get("response", {}).get("choices", [{}])[0].get("message", {}) if req else {}
        state = _text(req["messages"][-1]["content"]) if req else ""
        plan = _plan_block(state)
        active = next((l.split("active", 1)[1].strip() for l in plan.splitlines() if " active " in l), "")
        clock = f"turn {a['turn']} / {len(acts)} · sim {int(a['t']) // 60}:{int(a['t']) % 60:02d} · video at {speed}x"
        chat.msgs.append(Msg("harness", "exec", f"TASK STATE, turn {a['turn']}" + (f" · working on: {active}" if active else ""), dim=True))
        words = (reply.get("content") or a.get("reasoning") or "").strip()
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in a["args"].items())
        chat.msgs.append(Msg("exec", "harness" if a["tool"] in HARNESS_TOOLS else "robot", words, f"{a['tool']}({args})"))
        seg = frames[starts[i]:max(starts[i] + 1, starts[i + 1])] or [frames[-1]]
        need = max(len(seg), int(min_turn * FPS))
        cut = max(1, int(need * 0.55))  # the call shows first; the result when the turn's motion is done
        ck = checks.get(a["turn"])
        res = a["result"]
        for k in range(need):
            if k == cut:
                if ck:
                    chat.msgs.append(Msg("harness", "judge", f"does “{ck['condition']}” hold? (with the simulator's state)", dim=True))
                    chat.msgs.append(Msg("judge", "harness", "", ck.get("judge_reply") or ""))
                chat.msgs.append(Msg("harness" if a["tool"] in HARNESS_TOOLS else "robot", "exec", "",
                                     ("ok  " if res["success"] else f"{res['error']}  ") + res["message"]))  # fmt: skip
            fr = seg[min(k, len(seg) - 1)]
            emit(_left(fr, plan, clock, label), chat.image(title))
    gt = (end.get("ground_truth") or {}).get("success")
    chat.msgs.append(Msg("harness", None, f"{end.get('outcome')}: {end.get('summary')}", f"ground truth: {'SUCCESS' if gt else 'FAILURE'}"))
    emit(_left(frames[-1], plan, "the end", label), chat.image(title), 5 * FPS)
    proc.stdin.close()
    proc.wait()
    return Path(out or run / "chat_video.mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a run's video with its conversation beside it.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-turn", type=float, default=2.5, help="seconds each turn stays on screen at least")
    parser.add_argument("--speed", type=int, default=2, help="the speed the run's video was recorded at (run_agent --video)")
    args = parser.parse_args()
    print(f"wrote {render(args.run_dir, args.min_turn, args.speed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
