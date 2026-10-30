<!-- What the researcher's brief says about this robot ({ROBOT} in labs/briefs/*). -->
- **Robot.** A mobile Franka Panda arm with the Franka Hand (a parallel gripper that opens to
  77 mm), on a holonomic base. It is the simulated twin of a real robot: the same arm and hand,
  and cameras where the real ones are.
- **The hand.** In the flange frame (`docs/robot_api.py`), the fingertip pads' centers are
  0.093 m out along the flange's +z (the pads span 0.085–0.101 m), and the fingers close along
  the direction at −45° in the flange's x-y plane.
- **Sensors.** Two RGB-D cameras (640x480): `pole`, on a mast behind the base looking over the
  arm, and `wrist`, on the hand. Each comes with intrinsics and its pose. You also get the
  arm's joint state, the flange pose, the gripper and the base odometry. That is what the real
  robot has, and it is all the skill gets.
