You are controlling a robot arm with a two-finger parallel gripper (Robotiq 2F-85). Your job is to plan the gripper's whole motion to pick up the cup, as a list of waypoints, by marking points on plots of the scene.

THE PLOTS
Two plots are attached: first the TOP VIEW, then the SIDE VIEW.
- Both plots are orthographic (no perspective) and use the robot's own coordinate frame, in meters: x points forward, away from the robot (the robot is to the left of every plot, see the "robot" arrow); y points to the robot's left; z points up, measured from the floor.
- In both plots the horizontal axis is x. The vertical axis is y in the top view (looking straight down) and z in the side view (looking along +y, so the robot's right is toward you).
- Both plots have the same scale and equal aspect: a thin grid line every 1 cm, a thick labeled one every 5 cm.
- {appearance_text}
- In the side view the tan band is the counter the cup stands on.

THE GRIPPER
The panel at the right of each plot shows the gripper drawn to the same scale as the plot, approaching downward. Its two fingers close toward each other along the line between them; fully open they are 85 mm apart. It cannot open wider than that.
A gripper pose is given by three points: A and B, the two finger pads, and F, the center of the gripper's flange (the red bar at the top of the gripper in the panel), 15.6 cm behind the midpoint of A and B. The gripper's body lies along the line from the midpoint of A and B to F.
The gripper is now at waypoint 0, drawn in gray in both plots (A0, B0, F0): open, above the counter, between the robot and the cup.

WHAT TO DO
Pick up the cup. Give the waypoints the gripper should move through, in order, starting after waypoint 0. It moves in a straight line from each waypoint to the next (position and orientation change together along the way). Include the grasp, where the fingers are around part of the cup, and a lift afterwards that takes the cup clear of the counter. Use as few waypoints as you need, at most 8.
- For every waypoint mark A, B and F in both views; a point must be the same point in both views, so its x should match.
- For every waypoint say whether the gripper is "open" or "closed" when it gets there. It closes when it arrives at the first "closed" waypoint: that waypoint is the grasp, and the cup is held from then on.
- At the grasp, put A and B where the pads should be just before they close: on either side of the part of the cup you hold, each a few millimeters away from it, not touching it.
- On the way to the grasp nothing may hit the cup or the counter: no part of the gripper, and not the fingers either. Come in from a direction where the whole gripper stays clear.

Think it through first if you like. Then end your answer with exactly one JSON code block in this format:
```json
{{"waypoints": [{{"gripper": "open" or "closed", "top": {{"A": [x, y], "B": [x, y], "F": [x, y]}}, "side": {{"A": [x, z], "B": [x, z], "F": [x, z]}}}}, (more waypoints)]}}
```
Each value is in meters, read off that plot's axes.
