# Fingers: gripper lab run 2 `final` (2026-11-05)

The full-entry PETG HF finger with keyed TPU 85A pads from the agentic-skills gripper lab (run 2's
`generators/final`): `parts/` are the printed parts (millimetres), `collision/` the convex pieces the simulation
grips with (each with its material), `BUILD.md` how to print and mount them. In the gripper lab's finger frame: origin
at the centre of the carriage post's mounting face, x toward the other finger, y along the post's width, z down along
the finger. `robot.py` maps that onto the Franka Hand's finger body (origin on the hand axis 58.4 mm below the hand,
+y opening, +z approach): x_m = y_g, y_m = 15.5 mm - x_g, z_m = z_g - 5.5 mm.
