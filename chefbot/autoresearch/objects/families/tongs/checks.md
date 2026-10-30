# The physics checks

`./lab build` runs these on every asset (the code: `chefbot/autoresearch/objects/checks.py`; you may read it).
They run in MuJoCo with RoboCasa's physics settings (timestep 2 ms, elliptic friction cones, impratio 20,
noslip 4), in the lab frame: x along the arms toward the tips, y the opening (tip_a toward tip_b), z the hinge axis,
up. An asset **passes** when every check passes.

| check | what happens | passes when |
|---|---|---|
| structure | compiled alone; parts and sites looked up | the contract's bodies and sites exist; arms are not nested; collision budget kept; the origin within 1 cm of the bounding box centre; there are visual geoms |
| rest | held still on a stand, no gravity, 1 s | the arms don't interpenetrate; nothing moves; tip_b is on its side of tip_a |
| squeeze | forces on grip_a and grip_b, toward each other, ramped from 0 to the maximum over 3 s | the tips meet (closer than the contact gap, or touching near the tips); they never pass through each other; the arms don't sink into each other under the maximum force |
| release | the force removed | the tips are back at their rest gap within 1.5 s |
| hold | a test object between the tips, squeezed with a hand's force (1.5 x the closing force, or + 5 N, at most the hand's), then gravity on for 2 s | each object that fits in the opening is touched by both tips and moves less than 1 cm |
| counter | dropped onto a counter lying on its side (hinge axis up), 3 s | it comes to rest, stays on the counter, sinks in less than 2 mm once resting |
| hand_fit | | grip_a and grip_b are at most 70 mm apart at rest (the Franka Hand opens 77 mm) and closing takes at most 50 N |
| declared | measured vs your parameters | length, mass, rest tip gap and closing force each within tolerance |
| ranges | your parameters vs your ranges.yaml | every parameter inside its range; every range has a `why` |

Measured: `length_m` is the bounding box's extent along the arms (visual and collision geoms); `mass_kg` the whole
object's mass; `rest_tip_gap_m` the distance between the tip sites at rest; `closing_force_N` the force on each
grip site when the tips first meet.

Notes from building the lab:
- MuJoCo's contacts are points. A round object held between two flat pads by two point contacts can creep out
  under gravity; real tips hold with a contact patch. How you model the tips' contact (their shape, `condim`,
  `friction`, `solref`/`solimp`) is part of the job.
- MuJoCo's cylinder contacts are poor (few contact points); the lab's test objects are capsules, boxes, spheres.
- A mesh collides as its convex hull.
