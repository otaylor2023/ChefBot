# Sim to real: what the sims should model and don't yet

What the real FR3 showed that its simulations (the digital twin, `embodiments/twin`, and the kitchens it runs in,
agentic-skills-exploration's `chefbot_labs/environments`) don't reproduce. Each cost a real run. Skills trained in a sim
that lacks them carry the gap to the robot; model them in the sim's cameras and hand, with these numbers, before the
next labs.

## Depth cameras (TO MODEL: the sims render exact depth)

Measured 2026-11-06 (`onboard/calibration/depth_check.py`: a card's depth against its pose from its corners in the colour
image; frames as the skills get them, 640 x 480, depth aligned to colour):

| Camera | Distance | Depth reads | Off by |
|---|---|---|---|
| wrist D405 | 0.25 m | 1.001 x | +0.3 mm |
| wrist D405 | 0.68 m | 1.014 x | +9.7 mm |
| pole D455 | 0.35 m | 0.985 x | -5.3 mm |
| pole D455 | 0.69 m | 0.977 x | -16 mm |

- **Scale with distance**: the D405 reads long beyond its range (best 7-50 cm), the D455 short. Model it per camera as
  a scale growing with distance (a stereo camera's error grows with the square of the distance).
- **Small objects flattened at range**: the D405's 18 mm stereo baseline smooths small objects into what's behind them
  beyond ~0.5 m. An egg lying on the counter, 0.70 m away, read 1.5-2.2 cm tall (it is ~4.3), however much its mask was
  shrunk. The skill now looks again from 0.30 m (`kitchen_use/run1_v155_fr3` hold.py CLOSE_VIEW_M). Model: blur the
  depth over a window that grows with distance, or render it from a stereo pair at the real baseline.
- **Edges mix with the background**: a thin object's top edge (a standing chapstick, 1.5 cm wide) reads the counter
  behind it, dropping its top 1-1.5 cm even from 0.30 m. `hand_eye_check` takes heights from the silhouette instead.
  Model: at depth discontinuities, give pixels the far side's depth (or a mix) over a pixel or two.
- **Holes**: no depth on dark, shiny or thin parts (the pole camera found 0 depth points on a chapstick's mask once).

## The hand

- **The printed fingers' length**: on the real hand their ends are 0.148 m from the flange by touch
  (`onboard/calibration/tcp.py`, robot.yaml body hand_m; the pads' middle ~0.130 m), on the 27.25 mm camera holder the
  twin doesn't have. The twin's `PRINTED_PAD_CENTER_M` 0.091 is the design's figure without the holder; the real pads
  sit ~1.2 cm further out than the design plus the holder (0.118) would say.
- **Mass**: the twin gives each printed finger the stock finger's 15 g; they are ~40 g (the end-effector load is
  0.86 kg in Desk with them, 0.75 before).

## The arm and the cameras' poses

- The wrist camera's calibration from views behind the arm (the base card by the pole) was 2-2.5 cm off where the skills
  look; the joint calibration (`onboard/calibration/joint.py`, both cameras on one card on the counter) agrees within
  ~1 cm. A sim with exact camera poses never shows this: perturb them by ~1-2 cm and ~2 deg per episode.
- The FR3 sags a little when floating (joint 2's torque sensor is ~1.4 N m off), and the joint controller settles a few
  tenths of a degree short; neither is in the twin.
