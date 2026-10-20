You are controlling a robot arm with a two-finger parallel gripper (Robotiq 2F-85). Your job is to say where the gripper should go to {task_long}, by marking points on plots of the object.

THE PLOTS
{views_text}
- Both plots are orthographic (no perspective) and use the robot's own coordinate frame, in meters: x points forward, away from the robot (the robot is to the left of every plot, see the "robot" arrow); y points to the robot's left; z points up, measured from the floor.
- In both plots the horizontal axis is x. The vertical axis is y in the top view (looking straight down) and z in the side view (looking along +y, so the robot's right is toward you).
- Both plots have the same scale and equal aspect: a thin grid line every 1 cm, a thick labeled one every 5 cm.
- {appearance_text}
- In the side view the tan band is the counter the object stands on.

THE GRIPPER
The panel at the right of each plot shows the gripper drawn to the same scale as the plot, approaching downward. Its two fingers close toward each other along the line between them; fully open they are 85 mm apart. It cannot open wider than that.
{marks_text}

WHAT TO DO
{task_text}

Think it through first if you like: where the object is, which part the fingers should go around, and from which direction the gripper should come so the rest of it stays clear of the object and the counter. Then end your answer with exactly one JSON code block in this format:
{format_text}
