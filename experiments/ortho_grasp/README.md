# Ortho-view grasp marking (v1)

Can a VLM give a usable grasp by marking points on orthographic plots of a cup?
Spec: `ORTHO_GRASP_EXPERIMENT.md` at the repo root. Uses `chefbot/` read-only.

```
# one config, every stage (batch: submit, then run again later with --stage collect,evaluate)
python -m experiments.ortho_grasp.run --config experiments/ortho_grasp/configs/base.yaml
python -m experiments.ortho_grasp.run --config experiments/ortho_grasp/configs/base.yaml --stage collect,evaluate

# small synchronous runs (prompt tests): full price, no bucket needed
python -m experiments.ortho_grasp.run --config experiments/ortho_grasp/configs/base.yaml --scenes 0-2 --repeats 1 --mode sync

# execution in the simulator, then the report
python -m experiments.ortho_grasp.run --config experiments/ortho_grasp/configs/r_gt.yaml --stage e3
python -m experiments.ortho_grasp.report        # results/report.html, results/summary.csv

python -m pytest experiments/ortho_grasp/tests  # offline tests: no model, no simulator
```

Configs: `configs/base.yaml` holds every axis; each other config overrides one.
`experiment.yaml` is the kitchen, scenes and results folder; `models.yaml` the model ids.
OpenRouter batches need `gcs_bucket` (a public-read bucket: their batch API takes images only by URL).

Choices the spec left open, or where it had to bend:
- Robot frame = the mobile base frame (origin on the floor under the base center, x forward, z up).
- The arm is folded back while observing: stowed or at ready, it hides the cup from the shoulder camera.
- Cameras render at 960 x 720 (the env default 640 x 480 gives a very sparse partial cloud).
- Cup window 0.45-0.55 m in front of the arm base: deeper, a top-down pre-grasp puts the wrist into the wall cabinets.
- T3 plots use a 30 cm margin: the pre-grasp's flange is 25.6 cm above the grasp.
- R-GT is a ground-truth rim grasp computed from the cup mesh. The repo proposer's rim grasp
  assumes a container wider than the gripper and returns none for these cups; its top-down
  grasp on the 87 mm cup slides the fingers past an 87 mm rim.
- E1 `straddle` accepts one wall between the fingers (rim grasp) or both walls with both fingers
  outside the cup (body grasp).
- P2's approach search lets a finger touch the cloud (2 mm): marks on the surface are E1's to judge.
- E3 picks a collision-free IK solution for the pre-grasp (several seeds); E2's IK check is
  gripper-only, as specified.
- Scores are also given with a 5 mm contact tolerance per finger (`--stage rescore`: tol_*), since the v1
  prompt's "where the pads should be once the fingers close" made models mark pads on the surface,
  and with the flange corrected (fix_*: A and B kept, F moved to the nearest valid approach).
- Rendering goes through robosuite's own offscreen context: a second mujoco.Renderer in the same
  process renders with the wrong textures (the first V4 inputs were re-rendered and re-queried).

Full action chunks (follow-up): the model plans the whole pick from the gripper's current pose
(drawn in gray as waypoint 0), 1-8 waypoints; scored on the grasp (with contact tolerance), the
swept path and IK along it; `--e3` follows the waypoints in the sim as straight lines.

```
python -m experiments.ortho_grasp.chunk --model claude --scenes 0-9 --e3
```
