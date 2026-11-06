"""The whole-body controller every skill host runs (the sims' and the real robot's): wholebody.py ticks the base and
the arm toward a target; arm_solvers/ step the arm (qp, osc, legacy, inside a router); constraints.py is where the
arm may not go (a robot's body); kinematics.py the IK and frames. What it controls comes from embodiments/ (data).
"""
