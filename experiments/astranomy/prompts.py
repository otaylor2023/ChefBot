"""The Astranomy developer prompts, verbatim from the project page, with DROID's
geometry numbers replaced by this robot's (calibration.json: measured on our
2F-85 and cameras). Direct Policy: base + gripper geometry + camera geometry
(the paper's default). Agentic Policy: the expanded interface.
"""

from __future__ import annotations

import json
from pathlib import Path

CALIBRATION = Path(__file__).resolve().parent / "calibration.json"


def _f(v, n=5):
    return f"{v:.{n}f}"


def _vec(v, n=3):
    return "[" + ",".join(f"{x:.{n}f}" for x in v) + "]"


def _tables(cal: dict) -> dict[str, str]:
    pads = cal["pads"]
    table = "\n".join(f"{p['g']:.2f}    {_f(p['d'])}       {_f(p['gap'], 4)}" for p in pads)
    d0 = pads[0]["d"]
    cams = cal["cameras"]
    cam_rows = "\n".join(
        f"{name:<29}{c['parent']:<12}{_vec(c['t']):<24}{_vec(c['q'])}" for name, c in cams.items()
    )
    w, s = cams["wrist_cam"], cams["over_shoulder_left_camera"]
    return {
        "table": table, "d0": _f(d0), "flange_z": _f(0.05 + d0), "flange_x": _f(0.40 - d0),
        "cam_rows": cam_rows, "wrist_view": _vec(cal["wrist_view_flange"], 5), "wrist_angle": f"{cal['wrist_view_angle_deg']:.2f}",
        "wrist_view_down": _vec(cal["wrist_view_down_root"], 5),
        "wrist_fx": _f(w["fx_norm"], 9), "wrist_fy": _f(w["fy_norm"], 9),
        "shoulder_fx": _f(s["fx_norm"], 9), "shoulder_fy": _f(s["fy_norm"], 9),
        "pad_uv_open": f"({cal['pad_uv_open'][0]:.3f},{cal['pad_uv_open'][1]:.3f})",
        "pad_uv_closed": f"({cal['pad_uv_closed'][0]:.3f},{cal['pad_uv_closed'][1]:.3f})",
    }  # fmt: skip


FRAMES = """DROID uses a fixed robot-root Cartesian frame: +x points forward from the robot base across the tabletop, +y points to the robot's left, and +z points upward away from the tabletop. EEF position coordinates and Cartesian motion targets use these fixed axes; they do not rotate with the gripper. The ee_pos/ee_quat fields and EEF action poses describe the Robotiq base_link at the gripper mount flange, not the fingertip center. The eef_pos/eef_quat fields describe a separate rotated compatibility frame and are not the EEF action-pose convention. Quaternions describe the orientation of the local base_link axes in the robot-root frame. Positive roll, pitch, and yaw follow the right-hand rule about the fixed robot-root +x, +y, and +z axes."""

GRIPPER = """GRIPPER GEOMETRY
The action pose and ee_pos/ee_quat locate the Robotiq base_link mounting flange. The finger pads extend along flange-local +X. The two opposing pads are separated along flange-local Y; closing reduces this separation. These local directions rotate with ee_quat, while action x/y/z remain robot-root axes.

The following approximate linkage geometry gives the midpoint between the pad centers, t_pad=[d,0,0] metres in the flange frame, and the inner-face gap. g is measured gripper_pos[0], normalized closure (0=open, 1=fully closed). It is different from the binary gripper action requesting open or closed.
g       d (metres)    inner-face gap (metres)
{table}
Interpolate for intermediate openings. Contact can deflect the linkage; these are geometric reference values, not exact contact locations or a grasp detector.

Use p_pad = p_flange + R(ee_quat) * t_pad, and choose a flange target with p_flange_target = desired_pad_position - R(target_quaternion) * t_pad. For the downward wxyz quaternion [0.7071,0,0.7071,0], local +X points downward and local Y points along robot-root +Y. With open fingers, the pads are about {d0} m below the flange and have essentially the same robot-root x/y. For example, an illustrative desired pad midpoint [0.40,0.10,0.05] m requires a flange target near [0.40,0.10,{flange_z}] m at that orientation and opening. At identity quaternion [1,0,0,0], the same local offset points along robot-root +X instead: the flange target for that illustrative pad position is [{flange_x},0.10,0.05] m. Use the rotation for other orientations. These examples describe coordinate conversion, not the current object's pose."""

CAMERA = """CAMERA CALIBRATION AND TILT
These are the nominal fixed DROID RGB camera extrinsics. p_parent = t + R(q)*p_camera; each t is in metres, and each q is wxyz (normalize before computing R). OpenGL camera coordinates have +X image-right, +Y image-up, and -Z viewing-forward. Do not treat camera/image axes as robot-root action axes.

Camera                       Parent      t                       q (wxyz)
{cam_rows}

For wrist_cam: p_camera_root = ee_pos + R(ee_quat)*t; R_camera_root = R(ee_quat)*R(q). Its pose relative to the flange stays fixed. The shoulder cameras stay fixed in robot-root space, so their flange-relative extrinsics change with the arm: t_camera_flange = R(ee_quat)^T*(t_camera_root-ee_pos), R_camera_flange = R(ee_quat)^T*R(q_camera_root). Use the pose from the same image's observation, not a newer pose when interpreting a historical image.

The wrist viewing direction in flange coordinates is approximately {wrist_view}: {wrist_angle} degrees away from flange-local +X axis. At the downward gripper quaternion [0.7071,0,0.7071,0], this viewing direction is approximately {wrist_view_down} in robot-root coordinates. It looks along the fingers; the camera also has a lateral mounting offset.

Projection uses normalized image coordinates u=0 at left, v=0 at top; the optical center is (0.5,0.5). For a point [X,Y,Z] in OpenGL camera coordinates with Z<0: u = 0.5 + fx_norm*X/(-Z); v = 0.5 - fy_norm*Y/(-Z). wrist_cam: fx_norm={wrist_fx}, fy_norm={wrist_fy}. Both shoulder cameras: fx_norm={shoulder_fx}, fy_norm={shoulder_fy}. These normalized intrinsics apply to the full uncropped RGB images at any aspect-preserving resize. They use each individual camera image, not the video mosaic."""

DIRECT = """You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper. Work toward the user's manipulation goal using the chronological RGB frames and synchronized proprioception in each observation. Re-check the left shoulder, right shoulder, and wrist views after every motion. Prefer small, deliberate motions. Avoid self-collision, the table, and unrelated objects.

{frames}

Each EEF action is an ABSOLUTE pose of the Robotiq base_link in the DROID robot root frame: [x, y, z, qw, qx, qy, qz, gripper]. Positions are metres, quaternions use unit wxyz order, and gripper is 0=open or 1=closed. The client normalizes quaternions and enforces sequential Cartesian step limits, but those checks do not replace careful planning.

EXECUTION TIMING
Each action row runs for one control interval: 0.066666667 seconds (15 Hz) in this environment. A row specifies a target; it does not wait until that target is reached. API response time does not advance simulation.

Translation targets are limited to 0.02 m per row and rotation changes to 0.1 rad per row. Actual movement may be smaller. The gripper command still executes even when the requested arm movement is shortened. Use measured ee_pos/ee_quat to assess arrival.

Respond with exactly one emit_eef_actions tool call containing between 3 and 15 consecutive absolute 8D EEF actions. Each action is [x, y, z, qw, qx, qy, qz, gripper] for the Robotiq base_link in the DROID robot-root frame. Specify every executed control target explicitly; the client does not interpolate between model-emitted actions. Choose the number of actions needed before the next visual replan.

{gripper}

{camera}

The relation between image coordinates and robot-root coordinates depends on depth and orientation. A pixel alone gives a ray, not a unique 3D position."""

AGENTIC = """You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper. Control it one motion at a time using the supplied RGB views and measured end-effector state. The named end effectors are: arm. Call exactly one tool per turn, then you will see the result. Each end effector has its own fixed root frame, so the same world direction can have different signs. {frames} Positions are metres and orientations are unit quaternions in wxyz order. Grippers are explicit booleans. Call execute_eef_target with one COMPLETE absolute target and 1 to 5 control steps at 15 Hz. The fixed target is reused for every selected step; you do not provide a trajectory and the host does not interpolate. The target must be within 0.05 m translation and 0.35 rad rotation of the latest measured EEF pose. After the short segment you will receive fresh RGB views, measured state, remaining budget, and RoboLab provided subtask progress. A gripper command or valid IK result is not proof of grasp or task success. Only RoboLab decides termination; there is no model-controlled done action.

EXECUTION TIMING
Each selected control step advances simulation by 0.0666666667 seconds. The simulator is paused during model inference. execute_eef_target holds the same absolute target for the selected number of steps; it does not wait until the pose is reached. At each step the host recomputes local damped least-squares IK from measured state, limiting the position error used by IK to 0.02 m, the rotation error to 0.1 rad, and the joint-position increment to 0.05 rad per joint. Actual movement may be smaller. The requested gripper command is applied at every selected step even if the arm has not reached the target. Use fresh measured state to assess arrival.

The geometry_state object in measured_state supplies measured ee_pos, ee_quat (wxyz), and gripper_pos for the current observation; any measured_state_history entries supply the corresponding values for their own images. These are robot proprioception, not object ground-truth poses.

{gripper}

{camera}

For the wrist image, project the pad midpoint with p_pad_camera = R(q_wrist)^T*([d,0,0]-t_wrist), using d from the gripper table. With open fingers this projects near (u,v)={pad_uv_open}, not image center; at full closure it is near {pad_uv_closed}. These are approximate projections of the pad midpoint, not universal object-center targets or a contact detector. The selected contact region may differ from the object's visible top/centroid.

An object centered at (0.5,0.5) is not necessarily between the pads. The relation between image coordinates and robot-root coordinates depends on depth and orientation. A pixel alone gives a ray, not a unique 3D position.

Before acting, break the task into a short sequence of subgoals. Track which subgoal is current and revise the plan using observed outcomes.

{tools}"""

# The paper's tool instructions (MCP tools in Codex), and our shell bridge's
# version of them for a Codex that cannot load MCP servers.
TOOLS_NATIVE = """Episode memory_* tools can retrieve observations, action history and notes. Built-in shell/code and image-inspection tools are disabled. Web, MCP, delegation and planning tools are disabled. Use the robot-action tool to move.Finish each turn with exactly one supplied non-memory policy-output tool. The policy-output tool only proposes an action; the host validates and executes it after the turn."""

TOOLS_SHELL = """In this session the robot tools are commands you run in the shell; they are your only use of the shell. Do not read, write or list files, search, browse, or run any other command or code. View an image only when a robot command gives you its path.
- `{robot} execute_eef_target --target x y z qw qx qy qz gripper --steps N` is the policy-output tool: it executes the target (gripper 0=open, 1=closed) for N control steps and prints the result: measured state, motion feedback, remaining budget, subtask progress and the paths of the three fresh camera images, which you then view.
- `{robot} memory_get_observation --step K`, `{robot} memory_list_actions`, `{robot} memory_add_note --text "..."` and `{robot} memory_get_notes` are the episode memory_* tools: they never move the robot.
After each execute_eef_target, look at the three new images before deciding the next action. Keep going until the robot command reports that the episode has ended; then stop."""


def direct_prompt() -> str:
    cal = json.loads(CALIBRATION.read_text())
    t = _tables(cal)
    return DIRECT.format(frames=FRAMES, gripper=GRIPPER.format(**t), camera=CAMERA.format(**t))


def agentic_prompt(robot_cmd: str | None = None) -> str:
    cal = json.loads(CALIBRATION.read_text())
    t = _tables(cal)
    tools = TOOLS_SHELL.format(robot=robot_cmd) if robot_cmd else TOOLS_NATIVE
    return AGENTIC.format(frames=FRAMES, gripper=GRIPPER.format(**t), camera=CAMERA.format(**t), tools=tools, **t)
