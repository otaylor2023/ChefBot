<!-- What the researcher's brief says about this robot ({ROBOT} in labs/briefs/*). -->
- **Robot.** A Franka FR3 arm with the Franka Hand (a parallel gripper that opens to 80 mm) on a holonomic
  TidyBot++ base, the arm 0.335 m up at the base's center. It is a simulation of a real robot: the same arm, hand
  and base, and cameras where the real ones are.
- **The hand.** In the flange frame (`docs/robot_api.py`), the fingertip pads' centers are 0.103 m out along the
  flange's +z (the pads span 0.094–0.111 m), and the fingers close along the direction at +45° in the flange's x-y
  plane. The hand takes the real robot's two commands (libfranka's): `robot.gripper_move(width_m, speed_mps)`
  moves the fingers to a width (0–0.08 m), pushing as hard as it must (up to 70 N; if something stops them short it
  keeps pushing there, as hard as it can), and
  `robot.gripper_grasp(force_n, speed_mps)` closes until the fingers stop on something, then holds that force
  (0.1–70 N). Speeds up to 0.1 m/s. The hand finishes each command before taking the next. The force is not measured
  anywhere: you only get the fingers' width back. A row's gripper column is the shorthand: 1 a grasp at 20 N, 0 a move
  to full width, both at 0.05 m/s.
- **Sensors.** Two RGB-D cameras (640x480): `pole`, on a mast at the back of the base looking forward and down
  over the arm, and `wrist`, beside the hand looking along the fingers. Each comes with intrinsics and its pose.
  You also get the arm's joint state, the flange pose, the gripper (its last command and the fingers' width) and the
  base odometry.
  That is what the real robot has, and it is all the skill gets.
