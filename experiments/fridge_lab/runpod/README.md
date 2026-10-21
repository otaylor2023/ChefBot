# Fridge lab on RunPod (one RTX 4090)

## 1. The pod

- **Template:** RunPod PyTorch (Ubuntu 22.04, CUDA 12.x).
- **Disk:** a volume of at least 80 GB mounted at `/workspace`.
- **CPUs:** pick 16 or more vCPUs if you can, because each simulator worker uses about
  one core.

## 2. Code and keys

The pod needs the repo. Either push it to GitHub and clone it, or rsync it from your
Mac:

```bash
# on your Mac (the .env is copied separately, below)
rsync -az --exclude '.git' --exclude 'results' --exclude 'rollouts' --exclude 'runs' --exclude 'maps' \
  ~/Library/Mobile\ Documents/com~apple~CloudDocs/stanford/fall/cs329z/ChefBot/ root@<pod>:/workspace/ChefBot/
scp -P <port> ".../ChefBot/.env" root@<pod>:/workspace/ChefBot/.env
```

`experiments/fridge_lab/scenes.json` (the dev, val and test split) is in the repo. It
must be on the pod.

## 3. Setup (about 20–30 min, mostly the 10 GB of RoboCasa assets)

```bash
cd /workspace/ChefBot
bash experiments/fridge_lab/runpod/setup.sh
chmod 600 .env
su - researcher -c 'codex login --device-auth'    # open the link, log in with ChatGPT
bash experiments/fridge_lab/runpod/check.sh        # everything should pass
```

`check.sh` tells you whether Codex's own sandbox works in this container. Use the
`--codex-sandbox` it suggests.

## 4. The night

```bash
cd /workspace/ChefBot && source /workspace/fridge_lab/env.sh
tmux new -s research
/workspace/fridge_lab/venvs/sim/bin/python -m experiments.fridge_lab.research \
    --hours 8 --usage-cap 40 --model-usd 10 --jobs 8 \
    --codex-user researcher --codex-sandbox workspace-write    # or none (see check.sh)
# detach: Ctrl-b d
```

**Watch:**

- `tail -f /workspace/fridge_lab/private/research_run.log` shows the agent's messages,
  sessions and stops.
- `ls /workspace/fridge_lab/workspace/runs/` lists dev runs. `cat .../research_log.md`
  shows the agent's notes.
- `cat /workspace/fridge_lab/private/lab_state.json` lists val submissions and their
  scores.

The run stops at the time limit, when Codex reaches 40% of its weekly limit, or when
$10 of Gemini spend is reached. Each episode is also capped at $0.10, 120 s of think
time and 90 s of sim time.

## 5. Morning: the test set (run by us, once)

```bash
source /workspace/fridge_lab/env.sh
P=/workspace/fridge_lab/venvs/sim/bin/python
$P -m experiments.fridge_lab.evaluate --skill /workspace/fridge_lab/workspace/skills/final --split test --model-usd 3 --jobs 8
$P -m experiments.fridge_lab.evaluate --oracle --split test --jobs 8          # the ceiling
```

Results go to `/workspace/fridge_lab/eval/<name>/` (`summary.md`, `results.json`,
videos).

**Audit:**

- `grep -c "skill sandbox" .../eval/*/*/skill_log.txt` counts blocked skill actions.
- Codex's commands are in `private/codex_events.jsonl`. Grep them for `robocasa`,
  `scenes.json`, `private`, `layout00` and `style00`, and look for permission errors
  (attempts).
- Read `workspace/REPORT.md` and the git log.

## What is isolated, and how

| Who | Runs as | Can't read |
|---|---|---|
| Codex | `researcher` | `.env`, `scenes.json`, RoboCasa (including the test kitchens), the sim and vision venvs, `private/`, `eval/` |
| Skill | `fridgeskill`, under `bwrap` (no network) if the container allows it, and an audit hook | anything but its own files and the exchange folder |
| Lab, evaluator | root | everything; it holds the keys and the simulator |

If Codex runs with `--codex-sandbox none`, its shell has network access. The rules in
`AGENTS.md` forbid fetching test assets, and its commands are logged.
