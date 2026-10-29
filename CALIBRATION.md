# Camera calibration plan (real robot)

Status: plan only. The two RealSenses are not mounted yet; nothing here is built.

## What needs calibrating

| Camera | Mounted on | Solve | Result goes to |
|---|---|---|---|
| Wrist RealSense | the flange (moves with the arm) | eye-in-hand: camera in the flange frame | `envs/real/config.yaml` `cameras.wrist` (`parent: flange`) |
| Pole RealSense | the base / pole (fixed) | eye-to-hand: camera in the arm base frame, then the base frame | `cameras.pole` (`parent: base`) |

Intrinsics are not calibrated: RealSenses ship factory-calibrated, and `onboard/cameras.py`
reads them from the device. Depth is aligned to color on the robot, so one extrinsic per camera.

## Target

A ChArUco board (OpenCV `aruco.CharucoBoard`), not a single ArUco marker: dozens of corners
instead of four, usable when partly out of view, much better poses.

- Printed flat on something rigid (foam board or acrylic), matte.
- Square and marker size measured with calipers after printing: this sets the scale of everything.
- Board definition (dictionary, squares, sizes) kept in the calibration config, so detection matches the print.

## Procedure

### 1. Wrist camera (eye-in-hand)

1. Board fixed flat on the table, in reach.
2. The arm visits 15-25 poses looking at the board: varied tilts (about ±30 degrees about different axes)
   and distances (0.3-0.6 m). Rotation variety matters more than position variety; poses that only
   translate leave the solve ill-conditioned.
3. At each pose, after the arm settles: the joints (from `RealEnv` state) and one RGB frame.
4. Flange pose per sample: **our FR3 kinematics on the measured joints** (`ArmIK`, flange as TCP), not
   OpenSai's reported pose. Keeps the mount consistent with what the planner uses, and independent of
   the driver control-point question.
5. Board pose per sample: ChArUco corners -> `cv2.solvePnP` with the factory intrinsics. Drop frames
   with too few corners or high reprojection error.
6. Solve `cv2.calibrateHandEye` (try Tsai, Park, Daniilidis; they should agree within a few mm / tenths
   of a degree). Result: camera in the flange frame.

### 2. Pole camera (eye-to-hand)

Main method: board rigidly on the arm, in front of the pole camera.

1. Board on a 3D-printed bracket bolted to the flange (not held by the gripper: any slip ruins it).
2. The arm moves the board through 15-25 poses in the pole camera's view, same variety rules.
3. Per sample: flange pose (our kinematics) and board pose in the camera.
4. Solve `cv2.calibrateRobotWorldHandEye` (or `calibrateHandEye` with inverted flange poses).
   Result: camera in the arm base frame (and the board in the flange frame, as a by-product).
5. Convert to the base frame with `robot_real.yaml`'s `arm_mount` (which is itself still to be measured).

Cross-check (no bracket needed): once the wrist camera is calibrated, both cameras see the table board
at the same moment, and the chain arm base -> flange -> wrist camera -> board <- pole camera gives the
pole camera's pose. It inherits the wrist calibration's error, so it is the check, not the answer.

### 3. Verify

- Residuals: per-sample errors after the solve; anything over ~5 mm / 1 degree is suspect.
- Wrist: detect a marker, send the fingertip 2 cm above it, measure the miss with a ruler (target: < 5 mm).
- Pole: where it sees the board on the arm vs. where our kinematics put it, over a few new poses.
- Repeat the quick checks after anything bumps a camera; redo the calibration if they drift.

## How the poses get visited

- Default: slow joint moves through `RealEnv` (the arm's joint controller), the poses precomputed with our
  IK around the board, each one confirmed at the laptop before it runs. Needs the arm-motion bring-up
  first (joint moves), so calibration comes right after it.
- Alternative, no motion code: pose the arm by hand in a gravity-compensation mode and record. Only if
  OpenSai has a float controller (to check).

## Code (to build)

`chefbot/envs/real/calibrate.py`, on the laptop (needs `opencv-contrib-python` there, nothing new on the robot):

```sh
python -m chefbot.envs.real.calibrate wrist   # board on the table
python -m chefbot.envs.real.calibrate pole    # board on the flange bracket
python -m chefbot.envs.real.calibrate check   # the verification moves
```

- Collect: pose list -> confirm -> move -> settle -> frame + joints, saved under `runs/calibration/<time>/`
  so a solve can be rerun without the robot.
- Solve: from a saved run; prints residuals and the method comparison; writes the mount into
  `envs/real/config.yaml` only when asked.
- agentic-skills-exploration's `environments/real/calibrate_wrist.py` (fisheye intrinsics + hand-eye) has
  pose collection and solve code to adapt.

Can be built before the cameras are mounted: detection and solve, tested on synthetic data (a simulated
camera on our FR3 model looking at a rendered board, known mount, recovered within tolerance).

## Order

1. Mount the cameras for good; fill in serials (`onboard/config.yaml`).
2. Arm-motion bring-up (joint moves) — see the README's real-robot section.
3. Print and measure the board; make the flange bracket.
4. Wrist calibration, then pole (bracket), then the chain cross-check.
5. Verify; commit the mounts.
