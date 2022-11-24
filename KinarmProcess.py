##### Library for python simulation of Zahalak models.
###########
########### PARAMETERS
########### Return parameters for sims
# zahalak_86(): for fv figure 3b attempted reproduction. 
########### PLOTTING
# Return N from Zahalak States
# bond_distribution(Q0,Q1,Q2,parms)
# bond_ode_gif(odeOutput)
########### ODE FUNCTIONS
# odeZahalak(t,state,parm)

# optimization
import numpy as np
import casadi as ci #optimization
from casadi import SX as SX
import scipy.interpolate as sciinterp #used in fwd simulations
import scipy.integrate as integrate

# animations and plotting
from matplotlib import animation as animate
from matplotlib import pyplot as plt
from IPython import display

def hs2js(xy,l1,l2):
  x = xy[0]
  y = xy[1]
  c = ci.sqrt(x**2+y**2)
  gamma = ci.atan2(y,x)
  beta = ci.acos((l1**2+c**2-l2**2)/(2*l1*c))
  q1 = gamma - beta
  elb = ci.acos((l2**2+l1**2-c**2)/(2*l2*l1))
  q2 = ci.pi - (elb-q1)
  return ci.vertcat(q1,q2)

def tauGlob2Loc(tauGlob):
# function tauLoc = tauGlob2Loc(tauGlob)
    JacLoc2Glob = np.array([[1,0],[1,1]])
    return np.T(JacLoc2Glob) * tauGlob
  

def tauLoc2Glob(tauLoc):
#function tauGlob = tauLoc2Glob(tauLoc)        
    JacGlob2Loc = np.array([[1,0],[-1,1]])
    return np.T(JacGlob2Loc) * tauLoc

def invDynKinarmVirtualPower(theQ,theQDot,theQDDot,robTor,P):
 #function [tauGlobal,tauLocal,powers,meanMs,Ms] = invDynKinarmVirtualPower(...
                #theQ,theQDot,theQDDot,robTor,P)
# Compute inverse dynamics in global coordiante frame, 
# for added mass, need 3 new params in P: IHandMass (0?), mHandMass,
# cHandMass (axial distance from elbow joint to added mass)
# this uses equations of motion derived using virtual power.

  m1 = P.L1_M
  m2 = P.L2_M
  m3 = P.L3_M
  m4 = P.L4_M

  I1 = P.L1_I
  I2 = P.L2_I
  I3 = P.L3_I
  I4 = P.L4_I

  Imot1 = P.MOTOR1_I
  Imot2 = P.MOTOR2_I

  if hasattr(P,"IHandMass"):
      IHandMass = P.IHandMass
      mHandMass = P.mHandMass
      cHandMass = P.cHandMass
  else:
      IHandMass = 0
      mHandMass = 0
      cHandMass = 0


  l1 = P.L1_L
  l3 = P.L3_L
  cx1 = P.L1_C_AXIAL
  ca1 = P.L1_C_ANTERIOR
  cx2 = P.L2_C_AXIAL
  ca2 = P.L2_C_ANTERIOR
  cx3 = P.L3_C_AXIAL
  ca3 = P.L3_C_ANTERIOR
  cx4 = P.L4_C_AXIAL
  ca4 = P.L4_C_ANTERIOR
  Q25 = P.L2_L5_ANGLE

  #iloop =1:1:length(theQ)
  iloop = np.arange(0,theQ.shape[0])
  q1 = theQ[iloop,0]
  q2 = theQ[iloop,1]
  q1dot = theQDot[iloop,0]
  q2dot = theQDot[iloop,1]
  qddot1 = theQDDot[iloop,0]
  qddot2 = theQDDot[iloop,1]
  curRobTor1 = robTor[iloop,0]
  curRobTor2 = robTor[iloop,1]

  M11 = I1 + I4 + Imot1 + ca1**2*m1 + ca4**2*m4 + cx1**2*m1 + cx4**2*m4 + l1**2*m2 + l1**2*mHandMass
  M12 = cx2*l1*m2*ci.cos(q1 - q2) - ca4*l3*m4*ci.sin(Q25 + q1 - q2) + cHandMass*l1*mHandMass*ci.cos(q1 - q2) + ca2*l1*m2*ci.sin(q1 - q2) + cx4*l3*m4*ci.cos(Q25 + q1 - q2)
  #M21 = cx2*l1*m2*ci.cos(q1 - q2) - ca4*l3*m4*ci.sin(Q25 + q1 - q2) + cHandMass*l1*mHandMass*ci.cos(q1 - q2) + ca2*l1*m2*ci.sin(q1 - q2) + cx4*l3*m4*ci.cos(Q25 + q1 - q2),
  M22 = mHandMass*cHandMass**2 + m2*ca2**2 + m3*ca3**2 + m2*cx2**2 + m3*cx3**2 + m4*l3**2 + I2 + I3 + IHandMass + Imot2
  M21 = M12 #Kane/Lagrange derivations produce a symmetric mass matrix.

  meanMs = np.ndarray(2,2)
  meanMs[0,0] = np.mean(M11)
  meanMs[0,1] = np.mean(M12)
  meanMs[1,0] = np.mean(M21)
  meanMs[1,1] = np.mean(M22)
  #Ms = [repmat(M11,M12.shape[0],1),M12,M21,repmat(M22,length(M12),1)]

  F1 = -q2dot**2*(cx4*l3*m4*ci.sin(Q25 + q1 - q2) - ca2*l1*m2*ci.cos(q1 - q2) + cx2*l1*m2*ci.sin(q1 - q2) + cHandMass*l1*mHandMass*ci.sin(q1 - q2) + ca4*l3*m4*ci.cos(Q25 + q1 - q2))
  F2 = q1dot**2*(cx4*l3*m4*ci.sin(Q25 + q1 - q2) - ca2*l1*m2*ci.cos(q1 - q2) + cx2*l1*m2*ci.sin(q1 - q2) + cHandMass*l1*mHandMass*ci.sin(q1 - q2) + ca4*l3*m4*ci.cos(Q25 + q1 - q2))

  #matrix multiply manual for vectorization.
  #M ddq = F+[u1-u2u2]
  #DDQ = [qddot1qddot2]
  #M = [M11,M12M21,M22]
  tausNet1 = M11 * qddot1 + M12*qddot2 - F1 - curRobTor1 + curRobTor2
  tausNet2 = M21 * qddot1 + M22*qddot2 - F2 - curRobTor2

  tauGlobal0 = tausNet1 + tausNet2
  tauGlobal1 = tausNet2
  tauGlobal = ci.horzcat(tauGlobal0,tauGlobal1)
  tauG = ci.T(tauGlobal[iloop,:])
  tauL = tauGlob2Loc(tauG)
  tauLocal = ci.T(tauL) #because i've organized tau as row vecs.

  powerSho = theQDot[:,0]*tauGlobal[:,0]
  powerElb = theQDot[:,1]*tauGlobal[:,1] - theQDot[:,0]*tauGlobal[:,1]
  powers = [powerSho,powerElb]
  return tauLocal,tauGlobal, powers

# def deriveLocalEOM():
# syms m1 m2 m3 m4 I1 I2 I3 I4 l1 l3 cx1 ca1 cx2 ca2 cx3 ca3 cx4 ca4 Q25 g Imot1 Imot2 cHandMass mHandMass IHandMass real
# syms q1 q1dot q2 q2dot real
# %% DEFINE COMS
# % 20200226: EOM for kinarm with added mass
# % these EOMs include motor rotational inertias and endpoint masses.
# XYS = [cx1*cos(q1) - ca1*sin(q1)
#     cx1*sin(q1) + ca1*cos(q1)
#     l1*cos(q1) + cx2*cos(q1+q2) - ca2*sin(q1+q2)
#     l1*sin(q1) + cx2*sin(q1+q2) + ca2*cos(q1+q2)
#     cx3*cos(q1+q2 - Q25) - ca3*sin(q1+q2 - Q25)
#     cx3*sin(q1+q2 - Q25) + ca3*cos(q1+q2 - Q25)
#     l3*cos(q1+q2 - Q25) + cx4*cos(q1) - ca4*sin(q1)
#     l3*sin(q1+q2 - Q25) + cx4*sin(q1) + ca4*cos(q1)
#     0
#     0
#     0
#     0
#     l1*cos(q1) + cHandMass*cos(q1+q2)
#     l1*sin(q1) + cHandMass*sin(q1+q2)]

# % PHI_BODY_INERTIALFRAME: what is the orientation of the body in inertial reference frame?
# PHI_BODY_INERTIALFRAME = [q1q1+q2q1+q2q1q1q1+q2q1+q2]
# % what are the masses associated with each body?
# m=[m1m2m3m400mHandMass]
# % what are the inertias associated with each body?
# I=[I1 I2 I3 I4 Imot1 Imot2 IHandMass]
# % what are the DOF of the system?
# q = [q1q2]
# % what are the derivatives of the DOFs (2D is straightforward)
# qdot=[q1dotq2dot]

# n_bodies = length(m)
# n_q = length(q)
# n_dim = 2

# % LOOP.
# M = zeros(n_q,n_q)
# F = zeros(n_q,1)
# G = zeros(n_q,1)

# for i =1:n_bodies
#     XY_cur = XYS((i-1)*n_dim+1:(i)*n_dim,:)
#     PHI_cur = PHI_BODY_INERTIALFRAME(i)
#     JF = jacobian(XY_cur,q)
#     fprintf('Jacobian %i\n',i)
#     disp(JF)
#     JR = jacobian(PHI_cur,q)
#     sigma = jacobian(JF*qdot, q)*qdot
#     sigma_R = jacobian(JR*qdot,q)*qdot
#     meye = eye(n_dim,n_dim)*m(i)
#     M = M + JF' * meye * JF + ...
#         JR' * I(i) * JR
#     M = simplify(expand(M))
#     F = F - ( JF' * meye * sigma + JR' * I(i) * sigma_R) %minus is Remy convention.

#     %GRAVITY
#     % rotation matix: in case we want to reach in gravity...
#     gam = 0
#     A_IB = [[cos(gam)sin(gam)],[-sin(gam)cos(gam)]]
#     FG = m(i)*-g*A_IB'*[01]
#     G = G + JF'*FG
#     %/GRAVITY
# end
# F=simplify(expand(F))
# G=simplify(expand(G))
# out = struct
# out.F = F
# out.M = M
# out.G = G
