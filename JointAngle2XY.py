import casadi as ci
import numpy as np

def jointangle2xy(l1,l2,q1,q2):
  q1rad = q1*0.01745
  q2rad = q2*0.01745
  x = l1 * ci.cos(q1rad) + l2 * ci.cos(q1rad + q2rad)
  y = l1 * ci.sin(q1rad) + l2 * ci.sin(q1rad + q2rad)
  return np.array([x,y])
#%% 
import JointAngle2XY as JAXY
xystart = JAXY.jointangle2xy(.4,.3,62,5)
# %%
