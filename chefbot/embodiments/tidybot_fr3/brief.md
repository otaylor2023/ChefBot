<!-- What the researcher's brief says about this robot ({ROBOT} in labs/briefs/*). -->
- **Robot.** A Franka FR3 arm with the Franka Hand (a parallel gripper that opens to 83 mm) on a holonomic
  TidyBot++ base, the arm 0.335 m up at the base's center. It is a simulation of a real robot: the same arm, hand
  and base, and cameras where the real ones are.
- **The hand.** In the flange frame (`docs/robot_api.py`), the fingertip pads' centers are 0.103 m out along the
  flange's +z (the pads span 0.094–0.111 m), and the fingers close along the direction at +45° in the flange's x-y
  plane. The gripper command is continuous: 0 open, 1 closed, and g between them a gap between the pads of about
  0.083 − 0.080·g m. The fingers are a spring toward that gap (about 750 N of squeeze per m of gap, up to 70 N):
  command it narrower than what you hold and the hand squeezes harder.
- **Sensors.** Two RGB-D cameras (640x480): `pole`, on a mast at the back of the base looking forward and down
  over the arm, and `wrist`, beside the hand looking along the fingers. Each comes with intrinsics and its pose.
  You also get the arm's joint state, the flange pose, the gripper (command and opening) and the base odometry.
  That is what the real robot has, and it is all the skill gets.
