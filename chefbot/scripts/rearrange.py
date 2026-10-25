"""After exploring, move objects around, and score the map the agent keeps.

    python -m chefbot.scripts.rearrange                                      # symbolic: explore, then 6 moves
    python -m chefbot.scripts.rearrange --env robocasa --moves 4             # RoboCasa layout 1, ground-truth detect
    python -m chefbot.scripts.rearrange --env robocasa --map runs/<explore run>/explored_map.yaml

The agent starts from the explored map (the fixtures it was given and the
objects it found): the one in --map, or, without it, one made first by the
scripted explorer in a run of its own. A scripted policy (no LLM) then picks
objects up, drives somewhere else with them and puts them down on random
spots of other surfaces (chefbot/explore/rearrange.py). The agent's tools keep
its map: a pick takes the object off it, a place puts it back where it was
put, a detect moves it to where it is seen.

Only after the run is the true map made, and the scorer reads it:
runs/<run>/score.json has the pick and place success rates, every move, and
the agent's final map scored against the true one (map_after), next to the
explored map scored against it (map_stale: what a map that never followed the
moves would score).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

import yaml

from chefbot.agent.config import AgentConfig, LoopConfig
from chefbot.agent.loop.context import system_prompt
from chefbot.agent.loop.runner import AgentRunner, AgentSession
from chefbot.agent.plan.model import Plan
from chefbot.agent.world.map import WorldMap
from chefbot.explore.output import explored_map, save
from chefbot.explore.rearrange import RearrangeChecker, RearrangeObserver, ScriptedRearranger, initial_map
from chefbot.explore.score import describe, score
from chefbot.scripts import explore

REPO = Path(__file__).resolve().parents[2]
TOOLS = (*explore.TOOLS, "pick", "place", "holding")
PLAN = {
    "goal": "Objects moved to other surfaces, and the map keeping track of them",
    "steps": [{"id": "s1", "goal": "Move objects from where they are to other surfaces",
               "done_when": "the moves are done"}],
}  # fmt: skip
# Objects behind a door are left where they are: the policy does not open doors.
SKIP_IN = ("fridge", "microwave", "dishwasher", "oven", "cabinet", "drawer", "toaster oven")


def run(env: str = "symbolic", moves: int = 6, seed: int = 0, explored: Path | None = None,
        detector: str = "groundtruth", task: Path = explore.TASK, out: Path = REPO / "runs", noise: str = "off",
        quiet: bool = False, layout: int | None = None, style: int | None = None, detour: bool = True,
        look_after_place: bool = True, llm=None, video_speed: int = 0) -> dict:  # fmt: skip
    """`llm`: a policy in place of ScriptedRearranger (tests); it must have `.moves`.
    video_speed > 0 (robocasa): record runs/<run>/rearrange.mp4, keeping every Nth sim tick."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = out / (f"{stamp}_rearrange_{env}_s{seed}" + (f"_layout{layout:03d}" if layout else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    if explored is None:  # explore first, in a run of its own
        found = explore.run(env, "scripted", detector, seed, task, run_dir / "explore", quiet=True, layout=layout, style=style)
        explored = Path(found["run_dir"]) / "explored_map.yaml"
        if not quiet:
            print(f"explored: {found['found']}/{found['total']} objects ({found['run_dir']})")
    explored_dict = yaml.safe_load(Path(explored).read_text())

    world = (explore.symbolic_world(task, seed, noise) if env == "symbolic"
             else explore.robocasa_world(task, seed, run_dir, detector, layout, style))  # fmt: skip
    start = initial_map(world.start, explored_dict)
    (run_dir / "start_map.yaml").write_text(yaml.safe_dump(start, sort_keys=False, default_flow_style=None))
    world_map = WorldMap.from_dict(start)
    policy = llm or ScriptedRearranger(WorldMap.from_dict(start), moves, seed, skip_in=SKIP_IN, detour=detour,
                                       look_after_place=look_after_place)  # fmt: skip
    vocab = yaml.safe_load((explore.EXPLORE / "vocab.yaml").read_text())
    agent_config = dataclasses.replace(
        AgentConfig.load(), default_vocab=tuple(dict.fromkeys([*vocab["parts"], *vocab["objects"]])),
        search_default_vocab=False,
    )  # fmt: skip
    config = dataclasses.replace(LoopConfig.load(), max_turns=max(200, 40 * moves))
    session = AgentSession.build(
        Plan.from_dict(PLAN), world.robot, world.detector, world_map, world.clock, run_dir,
        RearrangeChecker(), RearrangeObserver(policy), config, agent_config,
        operator=lambda reason: "continue", system_prompt=system_prompt(set(TOOLS)), tools_only=TOOLS,
    )  # fmt: skip
    session.log.start(task="rearrange", task_file=str(task), seed=seed, moves=moves, explored_map=str(explored),
                      models={"agent": "scripted"}, tools=[s["function"]["name"] for s in session.registry.schemas],
                      **world.info)  # fmt: skip

    recorder = None
    if video_speed > 0:
        if world.stack is None:
            raise SystemExit("--video needs --env robocasa")
        from chefbot.scripts.record_rollout import Recorder

        # Renders the sim for the video only; the map panels draw the agent's own map as it changes.
        recorder = Recorder(world.stack, session.registry, WorldMap.from_dict(world.start), run_dir / "rearrange.mp4",
                            video_speed, semantic=lambda: explored_map(world.start, world_map)["objects"])  # fmt: skip

    def call_text(tool: str, args: dict) -> str:
        return f"{tool}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"

    def before_action(s: AgentSession, tool: str, args: dict, reasoning) -> None:
        move = policy.moves[-1] if policy.moves else None
        what = f"move {len(policy.moves)}: {move.object} from {move.source} to {move.target}" if move else ""
        recorder.caption = (f"> {call_text(tool, args)}", what)

    def after_action(s: AgentSession, tool: str, args: dict, result, reasoning) -> None:
        if recorder is not None:
            summary = f"{'ok' if result.success else 'FAILED: ' + str(result.error)}  {result.message}".strip()
            recorder.caption = (recorder.caption[0], summary if len(summary) <= 150 else summary[:147] + "...")
            if (result.data or {}).get("status") != "moving":
                recorder.hold(1.5 if tool in ("pick", "place") else 0.8)
        if not quiet:
            mark = "ok " if result.success else f"ERR {result.error}"
            call = call_text(tool, args)
            print(f"{s.turn:>4} {s.clock.now():>6.0f}s  {call:<48} {mark}  {result.message[:90]}", flush=True)

    try:
        if recorder is not None:
            recorder.caption = (f"moving things around {world.name}, starting from the map explored with "
                                f"{explored_by(explored)}", "")
            recorder.hold(1.0)
        outcome = AgentRunner(session, policy, after_action=after_action,
                              before_action=before_action if recorder is not None else None).run()  # fmt: skip
        final = explored_map(world.start, world_map)
        save(final, run_dir / "final_map.yaml")
        # The run is over: only now is the truth made, and only the scorer reads it.
        truth = world.truth()
    finally:
        if recorder is not None:
            world.robot.stop()
            recorder.close()
        world.close()
    save(truth, run_dir / "true_map.yaml")
    result = summarize(policy.moves, final, explored_dict, truth)
    result.update(outcome=list(outcome), run_dir=str(run_dir), explored_map=str(explored), explored_by=explored_by(explored))
    (run_dir / "score.json").write_text(json.dumps(result, indent=2))
    if not quiet:
        print("\n" + report(result) + f"\nrun: {run_dir}")
    return result


def explored_by(explored: Path) -> str:
    """Which detector built the explored map, from its run's start.json (when there is one)."""
    start = Path(explored).parent / "start.json"
    try:
        return json.loads(start.read_text()).get("detector") or "?"
    except (OSError, ValueError):
        return "?"


def summarize(moves, final: dict, explored: dict, truth: dict) -> dict:
    """Pick and place rates, each move, and the agent's map against the truth after the moves."""
    after = score(final, truth)
    rows = {r.get("found_as"): r for r in after["objects"] if r.get("found_as")}
    picks = [m for m in moves if m.pick_attempts]
    placed = [m for m in moves if m.picked]
    last = {m.object: i for i, m in enumerate(moves)}  # the map is scored once, after an object's last move
    out_moves = []
    for i, m in enumerate(moves):
        row = rows.get(m.object)
        if last[m.object] != i:
            out_moves.append({**m.to_dict(), "map_status": "moved again", "map_error_m": None, "map_problems": None})
            continue
        out_moves.append({**m.to_dict(), "map_status": row["status"] if row else "not on the map",
                          "map_error_m": row.get("error_m") if row else None,
                          "map_problems": row.get("problems") if row else None})  # fmt: skip
    return {
        "moves": out_moves,
        "picks": {"tried": len(picks), "ok": sum(m.picked for m in picks),
                  "first_try": sum(m.picked and m.pick_attempts == 1 for m in picks)},
        "places": {"tried": len(placed), "ok": sum(m.placed for m in placed), "dropped": sum(m.dropped for m in placed)},
        # Objects moved (put down at least once) that the map has where they truly are at the end.
        "moved_tracked": sum(r["map_status"] == "ok" for r in out_moves if r["map_status"] != "moved again"
                             and any(m.placed for m in moves if m.object == r["object"])),
        "moved_objects": len({m.object for m in moves if m.placed}),
        "map_after": after,
        "map_stale": score(explored, truth),
    }  # fmt: skip


def report(result: dict) -> str:
    p, q = result["picks"], result["places"]
    lines = [f"picks: {p['ok']}/{p['tried']} ({p['first_try']} first try)   places: {q['ok']}/{q['tried']}"
             + (f" ({q['dropped']} dropped)" if q["dropped"] else ""),
             f"moved objects where the map says: {result['moved_tracked']}/{result['moved_objects']}"]  # fmt: skip
    for m in result["moves"]:
        state = "moved" if m["placed"] else "not picked" if not m["picked"] else "not placed"
        err = (f", map {m['map_error_m']:.2f} m off" if m["map_error_m"] is not None
               else "" if m["map_status"] == "moved again" else f", map: {m['map_status']}")  # fmt: skip
        if m["map_problems"]:
            err += f" ({'; '.join(m['map_problems'])})"
        lines.append(f"  {m['object']}: {m['source']} -> {m['target']}: {state}{err}"
                     + (f" [{'; '.join(m['errors'])}]" if m["errors"] else ""))  # fmt: skip
    lines.append("after the moves, " + describe(result["map_after"]))
    s = result["map_stale"]
    lines.append(f"the explored map, never updated, would have {s['found']}/{s['total']} right")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=["symbolic", "robocasa"], default="symbolic")
    parser.add_argument("--moves", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", type=Path, help="explored_map.yaml to start from (default: explore first)")
    parser.add_argument("--detector", choices=["groundtruth", "vision", "owl"], default="groundtruth",
                        help="robocasa: what detect() is backed by (vision calls a paid VLM)")  # fmt: skip
    parser.add_argument("--task", type=Path, default=explore.TASK, help="the scene (default: explore_kitchen.yaml)")
    parser.add_argument("--layout", type=int)
    parser.add_argument("--style", type=int)
    parser.add_argument("--noise", choices=["off", "task", "on"], default="off", help="symbolic: injected failures")
    parser.add_argument("--no-detour", action="store_true", help="drive straight to where things go")
    parser.add_argument("--no-look", action="store_true", help="no detect after a place")
    parser.add_argument("--out", type=Path, default=REPO / "runs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true", help="robocasa: record runs/<run>/rearrange.mp4")
    parser.add_argument("--video-speed", type=int, default=2, help="keep every Nth sim tick (N x playback speed)")
    args = parser.parse_args()
    result = run(args.env, args.moves, args.seed, args.map, args.detector, args.task, args.out, args.noise,
                 args.quiet, args.layout, args.style, not args.no_detour, not args.no_look,
                 video_speed=args.video_speed if args.video else 0)  # fmt: skip
    if args.quiet:
        print(json.dumps({k: v for k, v in result.items() if k not in ("map_after", "map_stale")}, default=str))
    return 0 if result["map_after"]["perfect"] else 1


if __name__ == "__main__":
    sys.exit(main())
