# Starting a new Claude Code session (as of 2026-11-05)

Start Claude Code in `~/Documents/Stanford/cs329z-engineering-ai-agents/FINAL_PROJECT/Claude_Code/`. It reads
`CLAUDE.md` (its START HERE block points to the current notes) and its memory on its own. Pick the session type,
do the "before you start" items, then paste the first message.

## 1. Physical robot session (Mac)
Before you start:
- Relay remote in hand; the arm stays locked.
- Mac on SRC Wi-Fi if possible.
- Open the SSH connection (password once):
  ```
  ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-chefbot -o ControlPersist=8h chefbot
  ```
- Have ready: your list of the objects on the counters, and anything you changed physically (cables, camera, markers).

First message:
```
Read CLAUDE.md's START HERE, then mac_chefbot-movement/docs/kitchen-runbook.md and docs/memory-map-next.md.
Check who is using the robot, then bring up the base stack. Today: <your goal, e.g. one recorded pass on a new
layout, then go to three named objects>.
```

## 2. Offline memory-map / training session (Mac, heavy work on batcomputer)
Before you start, have ready:
- The object list (the detector's search terms and the answer key).
- What you moved during pass 3 and roughly when (it becomes `stream/p3_moved_unseen/MOVES.txt`).
- Whether a Gemini API key may be used.
- Whether to install ROCm PyTorch on batcomputer (it has an AMD GPU).

First message:
```
Read CLAUDE.md's START HERE and mac_chefbot-movement/docs/memory-map-next.md, especially "Read first". Work on
batcomputer in /mnt/d/Jack/Documents/Stanford/Course_Work/Engineering_AI_Agents/ (data in chefbot_data/2026-11-05/,
code in ChefBot_fresh on branch kitchen-nav). Start with gap 1: verify the pole camera chain against the Motive
markers in the infrared images.
```

## 3. Planner speed-up session (Mac, no robot)
First message:
```
Read CLAUDE.md's START HERE. In mac_chefbot-movement (branch kitchen-nav), speed up planning for 180-degree trips
across the island (~6 s now, Python A* repairs in robot/nav/heading_plan.py) and smooth the south-to-north path.
Benchmark in the simulator with real_nav park --mix ... --clearance 0.7 --turn-in-room as in kitchen-runbook.md.
```

## With your partner (not a Claude session)
- Review and merge PR #1, then PR #2.
- Commit the new pole calibration (`cameras.pole.T`, 2026-11-05) to the repo.
- Decide on the 0.5 m/s base cap (branch `speed-exp`).
- Arm review before any arm-on run.
