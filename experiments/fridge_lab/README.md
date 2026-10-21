# Fridge lab: autoresearch for a fast, cheap fridge-opening skill

GPT-6 Astra can open a fridge door with physics as the robot's policy (`experiments/astranomy`),
but it takes 5–16 minutes and up to dollars per episode. Here Codex, running GPT-6 Astra,
plays researcher instead. Overnight it writes and iterates on a **skill**: a program that opens
fridge doors on its own, calling only cheap models (Gemini Flash-Lite and Flash,
Robotics-ER, and open-vocabulary detection and SAM 2 on a GPU), and has to work on fridges it
has never seen. This follows the autoresearch loops of sooool and ENPIRE.

## The parts

| File | What |
|---|---|
| `kitchen.py` | Any RoboCasa kitchen's fridge, using the whole-body controller from `astranomy/fridge.py`. Finds the doors and handles, places a collision-free start pose relative to the door, and serves RGB-D plus proprioception |
| `oracle.py` | A privileged opener (ground-truth hinge and handle): the feasibility filter and the ceiling |
| `survey.py`, `splits.py` | Every fridge model run through the oracle, then the dev/val/test split (`scenes.json`) |
| `skill_api/fridge_robot.py` | The skill's side: the `Robot` API, and the process entry point with an audit-hook guard |
| `episode.py` | Runs one sandboxed skill episode and keeps its time: robot time = sim + think + model |
| `models.py`, `vision_server.py` | Metered Gemini (shared ledger, per-episode and total caps); OWLv2, Grounding DINO and SAM 2 server |
| `lab_server.py`, `lab_cli.py` | The researcher's `./lab`: dev runs with full logs, val as aggregate only (6 submissions), budget |
| `research.py`, `AGENTS.md`, `example_skill/` | The overnight Codex loop and its brief |
| `evaluate.py` | The final test-set evaluation of a frozen skill, and the oracle |
| `runpod/` | Setup, checks and a runbook for a 4090 pod |

## The split

RoboCasa's train and test kitchens (layouts and styles 11–60 versus 1–10) use disjoint fridge
models. The oracle opened 49 of the 50 models past 60° (`results/survey.json`). Of the train models,
24 are dev (31 doors) and 10 are val (14 doors, seeds 100–101). Test is RoboCasa's 15 test models
(20 doors, seeds 1000–1002).

## The rules, and how they're enforced

The skill sees only what a real robot has: RGB-D with calibration, proprioception, odometry.
It runs as its own user (on the pod), with no network, no subprocesses, no files beyond its own,
and no keys. All model calls go through the lab and are metered. The researcher (Codex) runs as
a user who can't read the keys, the split, RoboCasa (the test kitchens) or the lab's private
results. It sees dev results in full, val only as aggregates, and test never. `AGENTS.md` states
the rules (no peeking, no per-fridge lookup tables, honest reporting), and its commands are
logged for audit.
