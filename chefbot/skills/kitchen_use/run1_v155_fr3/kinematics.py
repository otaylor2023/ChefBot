"""Public FR3 kinematics; constants from Franka Robotics franka_description.
https://github.com/frankarobotics/franka_description/blob/main/robots/fr3/kinematics.yaml
https://github.com/frankarobotics/franka_description/blob/main/robots/fr3/joint_limits.yaml
No scene geometry or simulator state is used.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
ORIGINS=np.array([[0,0,.333,0],[0,0,0,-np.pi/2],[0,-.316,0,np.pi/2],[.0825,0,0,np.pi/2],[-.0825,.384,0,-np.pi/2],[0,0,0,np.pi/2],[.088,0,0,np.pi/2]])
LOW=np.array([-2.9007,-1.8361,-2.9007,-3.077,-2.8763,.4398,-3.0508])+.015
HIGH=np.array([2.9007,1.8361,2.9007,-.1169,2.8763,4.6216,3.0508])-.015
FIXED=[]
for x,y,z,a in ORIGINS:
 T=np.eye(4);T[:3,:3]=Rotation.from_euler('x',a).as_matrix();T[:3,3]=[x,y,z];FIXED.append(T)

def fk(q):
 T=np.eye(4)
 for A,a in zip(FIXED,q):
  c,s=np.cos(a),np.sin(a);R=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
  T=T@A@R
 T[:3,3]+=.107*T[:3,2]
 return T

def solve(F,seed,max_delta=.25):
 seed=np.array(seed)
 lower=np.maximum(LOW,seed-max_delta);upper=np.minimum(HIGH,seed+max_delta)
 if np.any(lower>=upper):return None
 def error(q):
  T=fk(q)
  return np.r_[T[:3,3]-F[:3,3],.2*Rotation.from_matrix(T[:3,:3]@F[:3,:3].T).as_rotvec(),.0005*(q-seed)]
 r=least_squares(error,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),max_nfev=35,ftol=1e-7,xtol=1e-7,gtol=1e-7)
 T=fk(r.x)
 if np.linalg.norm(T[:3,3]-F[:3,3])>.003 or Rotation.from_matrix(T[:3,:3]@F[:3,:3].T).magnitude()>.025:return None
 return r.x
