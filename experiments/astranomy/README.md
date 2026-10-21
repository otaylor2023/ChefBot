# Astranomy on the cup scenes

A reproduction of NVIDIA SRL's Astranomy (GPT-6 Astra as a zero-shot robot policy,
https://research.nvidia.com/labs/srl/projects/astranomy/) on the ortho_grasp cup scenes.

- `direct.py` — Direct Policy: stateless API calls (OpenRouter, `openai/gpt-6-astra`, medium reasoning);
  observations at t-14, t-7, t (three 384 x 216 views + measured state); one `emit_eef_actions`
  chunk of 3-15 absolute flange poses per call, each row one 15 Hz step (0.02 m / 0.1 rad per row).
- `agentic_codex.py` — Agentic Policy: a persistent Codex session (`gpt-6-astra`, medium) calling
  `execute_eef_target` (one target, 1-5 steps, within 0.05 m / 0.35 rad) and `memory_*` tools, with
  motion feedback after every action.
- `host.py` — the robot in DROID conventions (root = arm base, EEF = gripper mount flange, local +X
  along the fingers, Y across the pads) and the paper's controller (per-step IK clipped to 0.02 m,
  0.1 rad, 0.05 rad per joint; the sim is paused while the model thinks). Success: the cup's bottom
  5 cm above the counter, held.
- `prompts.py` — the paper's developer prompts, verbatim, with DROID's geometry tables replaced by
  this robot's (`calibration.json`, measured by `Host.calibration()`).

Differences from the paper: RoboCasa's two base-mounted cameras stand in for DROID's over-shoulder
pair (the wrist camera looks straight along the fingers, not tilted 18 degrees); the task is our
"Pick up the cup." with a lift check in place of RoboLab; and in Codex the tools are a `./robot`
shell command instead of MCP tools, because this Codex install's enterprise policy disables MCP
servers (the agent views the returned image paths with Codex's image viewer).

```
python -m experiments.astranomy.direct --scenes 0-4
python -m experiments.astranomy.agentic_codex --scenes 0-4
```

## Fridge (mobile base, physics)

`--task fridge`: open the main fridge door to at least 60 degrees by grasping its handle and pulling. There's no
scripted actuation and no handle lock. The robot starts about 0.8-1.0 m in front of the fridge (seeded). Every
target is absolute in a fixed odom frame (the robot root at the episode's start), and each one carries a base pose
(`fridge.py`): Direct rows are 11-D `[x,y,z,qw,qx,qy,qz,gripper,base_x,base_y,base_yaw]` (base limited to
0.02 m / 0.05 rad per row), and the Codex tool takes `--base bx by byaw` (within 0.10 m / 0.25 rad of the
measured base). Each control step drives the base toward its target and re-solves the arm against the measured
base. `fridge_oracle.py` is a scripted opener on the same controller: it shows the door opens by contact.

```
python -m experiments.astranomy.direct --task fridge --scenes 0-4
python -m experiments.astranomy.agentic_codex --task fridge --scenes 0-4
```
