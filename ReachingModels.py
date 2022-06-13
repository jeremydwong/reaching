#Reaching model classes that inherit from SimpleModel. 
import numpy as np
import casadi as ca
import scipy.io as io 
import scipy.integrate as integ
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import SimpleOpt as so

class PointMass(so.SimpleModel):
  g = 0
  DoF = 2
  NBodies = 1
  m = [1]
  kDamp = 0

  def inverseDynamics(self, q, qdot, qdotdot):
    return ca.vertcat(self.kDamp*qdot[0] + self.m[0]*qdotdot[0],
      self.kDamp*qdot[1] + self.m[0]*qdotdot[1])
  
  def implicitEOM(self, q, qdot, u, acc):
    return ca.vertcat(u[0]-self.kDamp*qdot[0] - self.m[0]*acc[0],
      u[1] - self.kDamp*qdot[1] - self.m[0]*acc[1])
  
  def implicitEOMQ(self, Q, acc):
    return self.implicitEOM(Q[0:2],Q[2:4],Q[4:6],acc)
  
  def kinematicJacobianInertias(self,q):
    return [np.array([[self.m[0],0],[0,self.m[0]]]), np.array([[self.m[0],0],[0,self.m[0]]])]
  
  def kinematicJacobianRotationalInertias(self):
    return [np.zeros([2,2]),np.zeros([2,2])]
  
  def xy2joints(self,xy):
    return xy
  
  def heightsMasses(self,q):
    return [q[1]]
  
  def jointPower(self,qdot, u):
    return ca.vertcat(qdot[0:1,:]*u[0:1,:], qdot[1:2,:]*u[1:2,:])
  
  def kinematicJacobianEndpoint(self, q):
    return np.array([[1,0],[0,1]])
  
  def handspeed(self, q, qdot):
    hvel = self.kinematicJacobianEndpoint(q) @ qdot
    htan = np.sqrt(hvel[0]**2+hvel[1]**2)
    return htan, hvel
  
  def setHandMass(self, inM):
    self.m = inM

class Kinarm(so.SimpleModel):
  DoF = 2
  NActuators = 2
  NBodies = 7
  g= 0.0
  parms = dict()

  m1 = 0
  m2 = 0
  m3 = 0
  m4 = 0

  I1 = 0
  I2 = 0
  I3 = 0
  I4 = 0

  l1 = 0
  l3 = 0
  cx1 = 0
  ca1 = 0
  cx1 = 0
  ca2 = 0
  cx3 = 0
  ca3 = 0
  cx4 = 0
  ca4 = 0
  Q25 = 0

  # def __init__ -> kinarmModel(fname)
  # input (optional): location of .mat param file.
  # output: class that has parameters set according to the file.
  def __init__(self,fname='/users/jeremy/Git/optreach/parameterFiles/paramsKinarmValidated80KgSubj.mat'):
    #fname='/users/jeremy/Git/optreach/parameterFiles/paramsKinarmValidated80KgSubj.mat'
    self.parms = io.loadmat(fname)
    self.m1 = self.parms["L1_M"].flatten()[0]
    self.m2 = self.parms["L2_M"].flatten()[0]
    self.m3 = self.parms["L3_M"].flatten()[0]
    self.m4 = self.parms["L4_M"].flatten()[0]
    self.mHandMass = 0.0
    self.m = [self.m1,self.m2,self.m3,self.m4,0.0,0.0,self.mHandMass]
    self.I1 = self.parms["L1_I"].flatten()[0]
    self.I2 = self.parms["L2_I"].flatten()[0]
    self.I3 = self.parms["L3_I"].flatten()[0]
    self.I4 = self.parms["L4_I"].flatten()[0]
    self.Imot1 = 0.001
    self.Imot2 = 0.001
    self.I = [self.I1,self.I2,self.I3,self.I4,self.Imot1,self.Imot2,0.0]

    self.l1 = self.parms["L1_L"].flatten()[0]
    self.l3 = self.parms["L3_L"].flatten()[0]
    self.l2 = self.parms["L2_L"] .flatten()[0]
    self.l = np.array([self.l1,self.l2])
    self.cx1 = self.parms["L1_C_AXIAL"].flatten()[0]
    self.ca1 = self.parms["L1_C_ANTERIOR"].flatten()[0]
    self.cx2 = self.parms["L2_C_AXIAL"].flatten()[0]
    self.ca2 = self.parms["L2_C_ANTERIOR"].flatten()[0]
    self.cx3 = self.parms["L3_C_AXIAL"].flatten()[0]
    self.ca3 = self.parms["L3_C_ANTERIOR"].flatten()[0]
    self.cx4 = self.parms["L4_C_AXIAL"].flatten()[0]
    self.ca4 = self.parms["L4_C_ANTERIOR"].flatten()[0]
    self.Q25 = self.parms["L2_L5_ANGLE"].flatten()[0]
    
    self.cHandMass = self.l2 #distance along the forearm where the mass is located.
    self.IHandMass = 0

  # def implicitJointEOM(self, q, qdot, torques, qdotdot):
  # inputs: states q, dqdt, torques u, accelerations acc.
  # outputs: errors in the contraint equations F - m*a = 0
  # note that we are using JOINT ANGLES (not external-reference-frame segment angles) 
  # which gives us the intuitive mass matrix
  def implicitEOM(self, q, qdot, u, qdotdot):
    M11 = self.I1 + self.I2 + self.I3 + self.I4 + self.IHandMass + self.Imot1 + self.Imot2 + self.ca1**2*self.m1 \
      + self.ca2**2*self.m2 + self.ca3**2*self.m3 + self.ca4**2*self.m4 + self.cx1**2*self.m1 + self.cx2**2*self.m2 \
      + self.cx3**2*self.m3 + self.cx4**2*self.m4 + self.cHandMass**2*self.mHandMass + self.l1**2*self.m2 \
      + self.l3**2*self.m4 + self.l1**2*self.mHandMass + 2*self.cx2*self.l1*self.m2*ca.cos(q[1]) \
      + 2*self.cHandMass*self.l1*self.mHandMass*ca.cos(q[1]) + 2*self.cx4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
      - 2*self.ca2*self.l1*self.m2*ca.sin(q[1]) - 2*self.ca4*self.l3*self.m4*ca.sin(self.Q25 - q[1])

    M12 = self.I2 + self.I3 + self.IHandMass + self.Imot2 + self.ca2**2*self.m2 + self.ca3**2*self.m3 \
       + self.cx2**2*self.m2 + self.cx3**2*self.m3 + self.cHandMass**2*self.mHandMass + self.l3**2*self.m4 \
         + self.cx2*self.l1*self.m2*ca.cos(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.cos(q[1]) \
           + self.cx4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) - self.ca2*self.l1*self.m2*ca.sin(q[1]) \
             - self.ca4*self.l3*self.m4*ca.sin(self.Q25 - q[1])

    M21 = M12 #symmetric mass matrix via virtual power

    M22 = self.mHandMass*self.cHandMass**2 + self.m2*self.ca2**2 + self.m3*self.ca3**2 + self.m2*self.cx2**2 \
      + self.m3*self.cx3**2 + self.m4*self.l3**2 + self.I2 + self.I3 + self.IHandMass + self.Imot2
    
    F1 =  qdot[1]*(2*qdot[0] + qdot[1])*(self.ca2*self.l1*self.m2*ca.cos(q[1]) - self.ca4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
       + self.cx2*self.l1*self.m2*ca.sin(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.sin(q[1]) - \
         self.cx4*self.l3*self.m4*ca.sin(self.Q25 - q[1]))
    F2 = -qdot[0]**2*(self.ca2*self.l1*self.m2*ca.cos(q[1]) - self.ca4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
      + self.cx2*self.l1*self.m2*ca.sin(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.sin(q[1]) - \
        self.cx4*self.l3*self.m4*ca.sin(self.Q25 - q[1]))
    
    #MassMat = np.array([[M11,M12],[M21,M22]])
    e1 = M11*qdotdot[0]+M12*qdotdot[1] - F1 - u[0] + u[1]
    e2 = M21*qdotdot[0]+M22*qdotdot[1] - F2 - u[1]
    return ca.vertcat(e1,e2)

  def inverseDynamics(self, q, qdot, qdotdot):
    M11 = self.I1 + self.I2 + self.I3 + self.I4 + self.IHandMass + self.Imot1 + self.Imot2 + self.ca1**2*self.m1 \
      + self.ca2**2*self.m2 + self.ca3**2*self.m3 + self.ca4**2*self.m4 + self.cx1**2*self.m1 + self.cx2**2*self.m2 \
      + self.cx3**2*self.m3 + self.cx4**2*self.m4 + self.cHandMass**2*self.mHandMass + self.l1**2*self.m2 \
      + self.l3**2*self.m4 + self.l1**2*self.mHandMass + 2*self.cx2*self.l1*self.m2*ca.cos(q[1]) \
      + 2*self.cHandMass*self.l1*self.mHandMass*ca.cos(q[1]) + 2*self.cx4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
      - 2*self.ca2*self.l1*self.m2*ca.sin(q[1]) - 2*self.ca4*self.l3*self.m4*ca.sin(self.Q25 - q[1])

    M12 = self.I2 + self.I3 + self.IHandMass + self.Imot2 + self.ca2**2*self.m2 + self.ca3**2*self.m3 \
       + self.cx2**2*self.m2 + self.cx3**2*self.m3 + self.cHandMass**2*self.mHandMass + self.l3**2*self.m4 \
         + self.cx2*self.l1*self.m2*ca.cos(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.cos(q[1]) \
           + self.cx4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) - self.ca2*self.l1*self.m2*ca.sin(q[1]) \
             - self.ca4*self.l3*self.m4*ca.sin(self.Q25 - q[1])

    M21 = M12 #symmetric mass matrix via virtual power

    M22 = self.mHandMass*self.cHandMass**2 + self.m2*self.ca2**2 + self.m3*self.ca3**2 + self.m2*self.cx2**2 \
      + self.m3*self.cx3**2 + self.m4*self.l3**2 + self.I2 + self.I3 + self.IHandMass + self.Imot2
    
    F1 =  qdot[1]*(2*qdot[0] + qdot[1])*(self.ca2*self.l1*self.m2*ca.cos(q[1]) - self.ca4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
       + self.cx2*self.l1*self.m2*ca.sin(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.sin(q[1]) - \
         self.cx4*self.l3*self.m4*ca.sin(self.Q25 - q[1]))
    F2 = -qdot[0]**2*(self.ca2*self.l1*self.m2*ca.cos(q[1]) - self.ca4*self.l3*self.m4*ca.cos(self.Q25 - q[1]) \
      + self.cx2*self.l1*self.m2*ca.sin(q[1]) + self.cHandMass*self.l1*self.mHandMass*ca.sin(q[1]) - \
        self.cx4*self.l3*self.m4*ca.sin(self.Q25 - q[1]))
    
    #MassMat = np.array([[M11,M12],[M21,M22]])
    u = np.zeros([2,1])
    u[1] = M21*qdotdot[0]+M22*qdotdot[1] - F2
    u[0] = M11*qdotdot[0]+M12*qdotdot[1] - F1 + u[1]
    return u

  # def implicitJointEOMQ(self, Q, acc):
  # inputs: stacked states Q [positions;velocities;torques!;torquerate]
  # outputs: errors in the 0 constraint equation F - m*a = 0
  def implicitEOMQ(self, Q, acc):
    return self.implicitEOM(Q[0:2],Q[2:4],Q[4:6],acc)
  
  # def kinematicJacobianInertias(self, q):
  # inputs: joint angles q
  # returns: a list of nq x nq matrices of the linear jacobians for inertias in this model.
  def kinematicJacobianInertias(self, q):
    q1 = q[0]
    q2 = q[1]
    j_1 = [[- self.ca1*ca.cos(q1) - self.cx1*ca.sin(q1), 0],[  self.cx1*ca.cos(q1) - self.ca1*ca.sin(q1), 0]]
    j_2 = [[- self.ca2*ca.cos(q1 + q2) - self.cx2*ca.sin(q1 + q2) - self.l1*ca.sin(q1), - self.ca2*ca.cos(q1 + q2) - self.cx2*ca.sin(q1 + q2)], \
      [  self.cx2*ca.cos(q1 + q2) - self.ca2*ca.sin(q1 + q2) + self.l1*ca.cos(q1),   self.cx2*ca.cos(q1 + q2) - self.ca2*ca.sin(q1 + q2)]]  
    j_3 = [[- self.ca3*ca.cos(q1 - self.Q25 + q2) - self.cx3*ca.sin(q1 - self.Q25 + q2), - self.ca3*ca.cos(q1 - self.Q25 + q2) - self.cx3*ca.sin(q1 - self.Q25 + q2)],\
      [  self.cx3*ca.cos(q1 - self.Q25 + q2) - self.ca3*ca.sin(q1 - self.Q25 + q2),   self.cx3*ca.cos(q1 - self.Q25 + q2) - self.ca3*ca.sin(q1 - self.Q25 + q2)]]
    j_4 = [[- self.l3*ca.sin(q1 - self.Q25 + q2) - self.ca4*ca.cos(q1) - self.cx4*ca.sin(q1), -self.l3*ca.sin(q1 - self.Q25 + q2)],\
      [  self.l3*ca.cos(q1 - self.Q25 + q2) + self.cx4*ca.cos(q1) - self.ca4*ca.sin(q1),  self.l3*ca.cos(q1 - self.Q25 + q2)]]
    j_mot1 = [[0, 0],\
      [0, 0]]
    j_mot2 = [[0, 0],\
      [0, 0]]
    j_handMass = [[- self.cHandMass*ca.sin(q1 + q2) - self.l1*ca.sin(q1), -self.cHandMass*ca.sin(q1 + q2)],\
      [  self.cHandMass*ca.cos(q1 + q2) + self.l1*ca.cos(q1),  self.cHandMass*ca.cos(q1 + q2)]]
    return [j_1, j_2, j_3, j_4, j_mot1, j_mot2, j_handMass]

  # def kinematicJacobianRotationalInertias(self):
  # input: ~
  # output: matrix to compute inertial-frame angular velocites. 
  def kinematicJacobianRotationalInertias(self):
    r1 = np.array([[1,0],[0,0]])
    r2 = np.array([[1,1],[0,0]])
    r3 = np.array([[1,1],[0,0]])
    r4 = np.array([[1,0],[0,0]])
    r5 = np.array([[1,0],[0,0]])
    r6 = np.array([[1,1],[0,0]])
    r7 = np.array([[1,1],[0,0]])
    return [r1,r2,r3,r4,r5,r6,r7]

  def joints2Endpoint(self, q):
      return ca.vertcat(self.l[0]*ca.cos(q[0])+self.l[1]*ca.cos(q[0]+q[1]),\
        self.l[0]*ca.sin(q[0])+self.l[1]*ca.sin(q[0]+q[1]))

  def heightsMasses(self,q):
    q1 = q[0]
    q2 = q[1]
    h1 = self.ca1*np.cos(q1) + self.cx1*np.sin(q1)
    h2 = self.ca2*np.cos(q1 + q2) + self.cx2*np.sin(q1 + q2) + self.l1*np.sin(q1)
    h3 = self.ca3*np.cos(q1 - self.Q25 + q2) + self.cx3*np.sin(q1 - self.Q25 + q2)
    h4 = self.l3*np.sin(q1 - self.Q25 + q2) + self.ca4*np.cos(q1) + self.cx4*np.sin(q1)
    h5 = 0
    h6 = 0
    h7 = self.cHandMass*np.sin(q1 + q2) + self.l1*np.sin(q1)
    return [h1,h2,h3,h4,h5,h6,h7]

  def jointPower(self,qdot, u):
    tauGlob = self.tauLoc2Glob(u)
    velGlob = self.joint2segment(qdot)
    return ca.vertcat(velGlob[0:1,:]*tauGlob[0:1,:], velGlob[1:2,:]*tauGlob[1:2,:] - velGlob[0:1,:]*tauGlob[1:2,:])

  def xy2joints(self, xy):
    x=xy[0]
    y=xy[1]
    l1 = self.l1
    l2 = self.l2
    c = np.sqrt(x**2+y**2)
    gamma = np.arctan2(y,x)
    beta = np.arccos((l1**2+c**2-l2**2)/(2*l1*c))
    q1 = gamma - beta
    elb = np.arccos((l2**2+l1**2-c**2)/(2*l2*l1))
    q2 = np.pi - (elb-q1)
    out = np.array([q1,q2-q1])
    if sum(np.isnan(out)) > 0:
      print("Warning! this hand position is unreachable given l1 and l2. Nans returned.")
    return out
  
  def kinematicJacobianEndpoint(self, q):
    l1 = self.l[0]
    l2 = self.l[1]
    kinJac = np.array([[ -l1*np.sin(q[0]) - l2*np.sin(q[0]+ q[1]), -l2*np.sin(q[0]+ q[1])],\
        [l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]),  l2*np.cos(q[0] + q[1])]])
    return kinJac

  def handspeed(self, q, qdot):
    hvel = self.kinematicJacobianEndpoint(q) @ qdot
    htan = np.sqrt(hvel[0]**2+hvel[1]**2)
    return htan, hvel

  #def tauGlob2Loc(tauGlob):
  # input: 'global' segment torques
  # output: 'local' joint torques
  def tauGlob2Loc(self,tauGlob):
  #tauLoc = tauGlob2Loc(tauGlob)
    JacLoc2Glob = np.array([[1,0],[1,1]])
    tauLoc = JacLoc2Glob.T @ tauGlob
    return tauLoc 
        
  # def tauLoc2Glob(tauLoc):
  # input: local joint torques
  # output: global segment torques
  def tauLoc2Glob(self,tauLoc):
    #function tauGlob = tauLoc2Glob(tauLoc)        
    JacGlob2Loc = np.array([[1,0],[-1,1]])
    tauGlob = JacGlob2Loc.T @ tauLoc
    return tauGlob

  def setHandMass(self, inM = 0.0):
    self.mHandMass = inM

  @staticmethod
  def segment2joint(angles):
    ang0 = angles[0:1,:]
    ang1 = angles[1:2,:] - angles[0:1,:]
    return ca.vertcat(ang0,ang1)
  
  @staticmethod
  def joint2segment(angles):
    ang0 = angles[0:1,:]
    ang1 = angles[1:2,:] + angles[0:1,:]

    return ca.vertcat(ang0,ang1)

class DoublePendulum(so.SimpleModel):
  DoF = 2
  NActuators = 2
  NBodies = 2
  l = np.array([.3,.3])
  d = np.array([0.15,0.15])
  I = np.array([1/12.0, 1/12.0, 0.0 ]) # hand mass default zero rotational inertia
  m = np.array([2.1, 1.65, 0.0])       # hand mass default zero mass
  g= 0.0

  def handspeed(self, q, qdot):
    hvel = self.kinematicJacobianEndpoint(q) @ qdot
    htan = np.sqrt(hvel[0]**2+hvel[1]**2)
    return htan, hvel
  
  def jointPower(self,qdot, u):
    return ca.vertcat(qdot[0:1,:]*u[0:1,:], qdot[1:2,:]*u[1:2,:]-qdot[0:1,:]*u[1:2,:])

  def kinematicJacobianEndpoint(self, q):
    l1 = self.l[0]
    l2 = self.l[1]
    kinJac = np.array([[ -l1*np.sin(q[0]), -l2*np.sin(q[1])],\
        [l1*np.cos(q[0]),  l2*np.cos(q[1])]])
    return kinJac
  
  # returns the global (inertial reference frame) velocities of the segments
  def kinematicJacobianInertias(self, q):
    j_uarm = np.array([[-self.d[0]*ca.sin(q[0]), 0],[self.d[0]*ca.cos(q[0]), 0]])
    j_larm = np.array([[-self.l[0]*ca.sin(q[0]), -self.d[1]*ca.sin(q[1])],[self.l[0]*ca.cos(q[0]),  self.d[1]*ca.cos(q[1])]])
    j_hand = np.array([[-self.l[0]*ca.sin(q[0]), -self.l[1]*ca.sin(q[1])],[ self.l[0]*ca.cos(q[0]),  self.l[1]*ca.cos(q[1])]])
    
    return [j_uarm, j_larm, j_hand]
  
  def joints2Endpoint(self, q):
      return ca.vertcat(self.l[0]*ca.cos(q[0])+self.l[1]*ca.cos(q[1]),\
        self.l[0]*ca.sin(q[0])+self.l[1]*ca.sin(q[1]))

  # returns the global (inertial reference frame) angular velocities of the segments
  def kinematicJacobianRotationalInertias(self):
    r0 = np.array([[1,0],[0,0]])
    r1 = np.array([[0,1],[0,0]])
    r2 = np.array([[0,0],[0,0]])
    return [r0,r1,r2]

  # returns the heights of the masses as a list
  def heightsMasses(self,theQ):
    g0 = self.d[0]*ca.sin(theQ[0])*self.m[0]*self.g
    g1 = self.l[0]*ca.sin(theQ[0]) + self.d[1]*ca.sin(theQ[1])*self.m[1]*self.g
    g2 = self.l[0]*ca.sin(theQ[0]) + self.l[1]*ca.sin(theQ[1])*self.m[2]*self.g
    
    return [g0,g1,g2]

  def setHandMass(self, inM = 0.0):
    return 0

  def xy2joints(self, xy):
    #function out = xy2joints(xy,lengths)
    # converts handspace x and y to joint angles. 
    # assumes shoulder at [0,0]!!
    # assumes solution is for the right arm (since you have to make a choice
    # about putting the arm somewhere)
    # output is two external-reference-frame angles. 
    x=xy[0]
    y=xy[1]
    l1 = self.l[0]
    l2 = self.l[1]
    c = np.sqrt(x**2+y**2)
    gamma = np.arctan2(y,x)
    beta = np.arccos((l1**2+c**2-l2**2)/(2*l1*c))
    q1 = gamma - beta
    elb = np.arccos((l2**2+l1**2-c**2)/(2*l2*l1))
    q2 = np.pi - (elb-q1)
    out = np.array([q1,q2])
    if sum(np.isnan(out)) > 0:
      print("Warning! this hand position is unreachable given l1 and l2. Nans returned.")
    return out

  # EOM DEFINITION. 
  def inverseDynamics(self,theQ,theQDot,theQDotDot):
    g = self.g
    l1 =self.l[0]
    l2 =self.l[1]
    m1 =self.m[0]
    m2 =self.m[1]
    m3 =self.m[2]
    d1 =self.d[0]
    d2 =self.d[1]
    I1 =self.I[0]
    I2 =self.I[1]

    q1 = theQ[0]
    q2 = theQ[1]
    q1dot = theQDot[0]
    q2dot = theQDot[1]
    acc1 = theQDotDot[0]
    acc2 = theQDotDot[1]

    F11 = -l1*q2dot**2 * ca.sin(q1 - q2)*(d2*m2 + l2*m3)
    F21 =  l1*q1dot**2 * ca.sin(q1 - q2)*(d2*m2 + l2*m3)
    G11 = -g*ca.cos(q1)*(d1*m1 + l1*m2 + l1*m3)
    G21 = -g*ca.cos(q2)*(d2*m2 + l2*m3)
    M11 = I1 + d1**2*m1 + l1**2*m2 + l1**2*m3 
    M12 = l1*ca.cos(q1 - q2)*(d2*m2 + l2*m3)
    M21 = l1*ca.cos(q1 - q2)*(d2*m2 + l2*m3)
    M22 = m2*d2**2 + m3*l2**2 + I2    

    tau2 = M21*acc1 + M22*acc2 - F21 - G21
    tau1 = M11*acc1 + M12*acc2 - F11 - G11 + tau2

    return np.array([[tau1],[tau2]])

  def implicitEOMQ(self,qin,accin):
    # m*a - F =0. 
    g = self.g
    l1 =self.l[0]
    l2 =self.l[1]
    m1 =self.m[0]
    m2 =self.m[1]
    m3 =self.m[2]
    d1 =self.d[0]
    d2 =self.d[1]
    I1 =self.I[0]
    I2 =self.I[1]

    q1 = qin[0]
    q2 = qin[1]
    q1dot = qin[2]
    q2dot = qin[3]
    u1i = qin[4]
    u2i = qin[5]

    F11 = -l1*q2dot**2 * ca.sin(q1 - q2)*(d2*m2 + l2*m3)
    F21 =  l1*q1dot**2 * ca.sin(q1 - q2)*(d2*m2 + l2*m3)
    G11 = -g*ca.cos(q1)*(d1*m1 + l1*m2 + l1*m3)
    G21 = -g*ca.cos(q2)*(d2*m2 + l2*m3)
    M11 = I1 + d1**2*m1 + l1**2*m2 + l1**2*m3 
    M12 = l1*ca.cos(q1 - q2)*(d2*m2 + l2*m3)
    M21 = l1*ca.cos(q1 - q2)*(d2*m2 + l2*m3)
    M22 = m2*d2**2 + m3*l2**2 + I2
    
    acc1 = accin[0]
    acc2 = accin[1]
    tau1 = u1i - u2i
    tau2 = u2i
    e1 = M11*acc1 + M12*acc2 - F11 - G11 - tau1
    e2 = M21*acc1 + M22*acc2 - F21 - G21 - tau2
    #eqs = MassMat @ accs - F - G - taus
    return ca.vertcat(e1,e2)

  
    