#Double Pendlum Class
from calendar import c
from http.cookies import SimpleCookie
from typing import Callable
import numpy as np
import casadi as ca
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pygsheets
import SimpleOpt as so

class Struct:
  def __init__(self, *args, prefix='arg'): # constructor
      self.prefix = prefix
      if len(args) == 0:
          self.i = 0
      else:
          i=0
          for arg in args:
              i+=1
              arg_str = prefix + str(i)
              # store arguments as attributes
              setattr(self, arg_str, arg) #self.arg1 = <value>
          self.i = i
  def add(self, name,val):
      self.i += 1
      setattr(self, name, val)

class doublePendulum:
  DoF = 2
  NActuators = 2
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
  
  def kinematicJacobianInertias(self, q):
    j_uarm = np.array([[-self.d[0]*ca.sin(q[0]), 0],[self.d[0]*ca.cos(q[0]), 0]])
    j_larm = np.array([[-self.l[0]*ca.sin(q[0]), -self.d[1]*ca.sin(q[1])],[self.l[0]*ca.cos(q[0]),  self.d[1]*ca.cos(q[1])]])
    j_hand = np.array([[-self.l[0]*ca.sin(q[0]), -self.l[1]*ca.sin(q[1])],[ self.l[0]*ca.cos(q[0]),  self.l[1]*ca.cos(q[1])]])
    
    return [j_uarm, j_larm, j_hand]
  
  def kinematicJacobianRotationalInertias(self,q):
    r0 = np.array([[1,0],[0,0]])
    r1 = np.array([[0,1],[0,0]])
    r2 = np.array([[0,0],[0,0]])
    return [r0,r1,r2]

  def heightsMasses(self,theQ):
    g0 = self.d[0]*ca.sin(theQ[0])*self.m[0]*self.g
    g1 = self.l[0]*ca.sin(theQ[0]) + self.d[1]*ca.sin(theQ[1])*self.m[1]*self.g
    g2 = self.l[0]*ca.sin(theQ[0]) + self.l[1]*ca.sin(theQ[1])*self.m[2]*self.g
    
    return np.array([g0,g1,g2])

  def energy(self, theQ, theQDot, theU, theT):
    
    nT = theQ.shape[1]
    
    # mecahnical power
    eDot_mech = self.jointPower(theQDot,theU)
    t = theT.reshape([1,-1])
    e_mechAll = integ.cumulative_trapezoid(eDot_mech,x = t,initial=0) #axis = -1 by default, the last axis [which is true], initial keeps shape same.
    e_mech = e_mechAll[0:1,:]+e_mechAll[1:2,:]
    
    # gravitational work
    e_g = np.zeros([1,theQ.shape[1]])
    for it in np.arange(0,theQ.shape[1]):
      heights = self.heightsMasses(theQ[:,it])
      for ih in np.arange(0,len(heights)):
        e_g[0,it] = e_g[0,it] + self.m[ih]*self.g*heights[ih]
    e_g = e_g - e_g[0] # subtract initial

    # kinetic energy
    e_k = np.zeros([1,theQ.shape[1]])
    for it in np.arange(0,theQ.shape[1]):
      linjac = self.kinematicJacobianInertias(theQ[:,it])
      rotjac = self.kinematicJacobianRotationalInertias(theQ[:,it])
      for ij in np.arange(0,len(linjac)):
        vt = linjac[ij] @ theQDot[:,it]
        angvt = rotjac[ij] @ theQDot[:,it]
        e_k[0,it] = e_k[0,it] + 1/2 * self.m[ij] * vt.T @ vt
        e_k[0,it] = e_k[0,it] + 1/2 * self.I[ij] * angvt.T @ angvt

    energyOut = so.SimpleModelEnergy()
    energyOut.e_g = e_g
    energyOut.e_k = e_k
    energyOut.e_mechAll = e_mechAll
    energyOut.e_mech = e_mech
    return energyOut

  @staticmethod 
  def xy2joints(xy,lengths):
    #function out = xy2joints(xy,lengths)
    # converts handspace x and y to joint angles. 
    # assumes shoulder at [0,0]!!
    # assumes solution is for the right arm (since you have to make a choice
    # about putting the arm somewhere)
    # output is two external-reference-frame angles. 
    x=xy[0]
    y=xy[1]
    l1 = lengths[0]
    l2 = lengths[1]
    c = np.sqrt(x**2+y**2)
    gamma = np.arctan2(y,x)
    beta = np.arccos((l1**2+c**2-l2**2)/(2*l1*c))
    q1 = gamma - beta
    elb = np.arccos((l2**2+l1**2-c**2)/(2*l2*l1))
    q2 = np.pi - (elb-q1)
    return np.array([q1,q2])

  # EOM DEFINITION. 
  def implicitEOM(self,qin,accin):
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

  def movementTimeOpt(self, theXYStart, theXYEnd, theN=100, theFRCoef = 8.5e-2, theTimeValuation = 1, theGeneratePlots = 1):
  # def movementTimeOpt(self, xystart, xyend, N=100, generate_plots = 1):
  # Trajectory optimization problem formulation, hermite simpson, implicit dynamics constraints means
  # We ask the solver to find (for example) accelerations and write the equations of motion implicitly. 
    
    # Create opti instance.
    opti = ca.Opti()

    # Opt variables for duration, state, and controls.  
    duration = opti.variable() #opti.variable()
    dt = duration/theN
    time = ca.linspace(0., duration, theN+1)  # Discretized time vector
    qAll = opti.variable(8, theN+1)
    # position
    Q = qAll[0:2,:]
    q1 = qAll[0, :]
    q2 = qAll[1, :]
    # velocity
    QDot = qAll[2:4,:]
    dq1 = qAll[2, :]
    dq2 = qAll[3, :]
    # force
    U = qAll[4:6,:]
    u1 = qAll[4, :]
    u2 = qAll[5, :]
    # force rate
    du = qAll[6:8,:]
    
    # optimal acc. a decision variable allowing implicit equations of motion.
    acc = opti.variable(2,theN+1)

    # Force rate
    uraterate = opti.variable(2, theN+1)

    # Calculus equation constraint
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete time

    so.HermiteSimpsonImplicit(opti,qd,self.implicitEOM,qAll,acc,uraterate,dt,[2,3])
    
    mechPower = self.jointPower(QDot,U)

    # CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
    # variables will be bounded between +/- Inf).
    opti.subject_to(opti.bounded(-10, Q[0,:], 10))
    opti.subject_to(opti.bounded(-10, Q[1,:], 10))

    ### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
    # # slack variables for power and force-rate-rate
    slackVars = opti.variable(6, theN+1)
    pPower = slackVars[0:2, :]
    pddu = slackVars[2:4,:]
    nddu = slackVars[4:6,:]
    
    # positive power constraints
    opti.subject_to(pPower[0,:] >= 0.) 
    opti.subject_to(pPower[1,:] >= 0.) 
    opti.subject_to(pPower[0,:] >= mechPower[0,:]) 
    opti.subject_to(pPower[1,:] >= mechPower[1,:]) 
    # fraterate constraints
    opti.subject_to(pddu[0,:] >= 0.)  
    opti.subject_to(pddu[1,:] >= 0.)  
    opti.subject_to(pddu[0,:] >= uraterate[0,:])  
    opti.subject_to(pddu[1,:] >= uraterate[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= uraterate[0,:]) 
    opti.subject_to(nddu[1,:] <= uraterate[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY)
    # Boundary constraints - periodic gait.
    qCON0 = doublePendulum.xy2joints(theXYStart,self.l)
    qCON1 = doublePendulum.xy2joints(theXYEnd,self.l)

    opti.subject_to(Q[0,0] == qCON0[0])
    opti.subject_to(Q[1,0] == qCON0[1])
    opti.subject_to(Q[0,-1] == qCON1[0])
    opti.subject_to(Q[1,-1] == qCON1[1])
    opti.subject_to(QDot[0,0] == 0.0)
    opti.subject_to(QDot[1,0] == 0.0)
    opti.subject_to(QDot[0,-1] == 0.0)
    opti.subject_to(QDot[1,-1] == 0.0)
    opti.subject_to(U[0,0] == 0.0)
    opti.subject_to(U[1,0] == 0.0)
    opti.subject_to(U[0,-1] == 0.0)
    opti.subject_to(U[1,-1] == 0.0)

    ### i don't understand why these are infeasible
    opti.subject_to(du[0,0] == 0.0)
    opti.subject_to(du[1,0] == 0.0)
    opti.subject_to(du[0,-1] == 0.0)
    opti.subject_to(du[1,-1] == 0.0)

    opti.subject_to(uraterate[0,0] == 0.0)
    opti.subject_to(uraterate[1,0] == 0.0)
    opti.subject_to(uraterate[0,-1] == 0.0)
    opti.subject_to(uraterate[1,-1] == 0.0)

    opti.subject_to(acc[0,0] == 0.0)
    opti.subject_to(acc[1,0] == 0.0)
    opti.subject_to(acc[0,-1] == 0.0)
    opti.subject_to(acc[1,-1] == 0.0)

    opti.subject_to(pddu[0,0] == 0.0)
    opti.subject_to(pddu[1,0] == 0.0)
    opti.subject_to(pddu[0,-1] == 0.0)
    opti.subject_to(pddu[1,-1] == 0.0)

    opti.subject_to(duration >=0.0)  # critical!
    opti.subject_to(duration <=20.0) # maybe unnecessary! =)
    
    ############################################################################################################################################
    ############## OBJECTIVE ############## 
    frCoef = theFRCoef
    timeValuation = theTimeValuation
    costTime = duration * timeValuation
    costWork = so.trapInt(time,pPower[0,:])+so.trapInt(time,pPower[1,:])
    costFR = frCoef * (so.trapInt(time,pddu[0,:]) + so.trapInt(time,pddu[1,:]) - so.trapInt(time,nddu[0,:]) - so.trapInt(time,nddu[1,:]))
    costJ = costTime + costWork + costFR
    # Set cost function
    opti.minimize(costJ)
    
    ############################################################################################################################################
    ############## GUESS ############## 
    q1guess = np.linspace(qCON0[0], qCON1[0], theN+1)
    q2guess = np.linspace(qCON0[1], qCON1[1], theN+1)
    opti.set_initial(q1, q1guess)
    opti.set_initial(q2, q2guess)
    opti.set_initial(duration,1.0)

    ############################################################################################################################################
    ############## Hyperparameters and solve ############## 
    maxIter = 200
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(time),opti.debug.value(q1),
          opti.debug.value(time), opti.debug.value(q2),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)
    sol = opti.solve()

    ############################################################################################################################################
    ############## Post optimization ############## 
    
    # Extract the optimal states and controls.
    optTraj = so.optTrajectories()
    optTraj.time = sol.value(time)
    optTraj.Q = sol.value(Q)
    optTraj.QDot = sol.value(QDot)
    optTraj.U = sol.value(U)
    optTraj.mechPower = sol.value(mechPower)
    optTraj.costJ = sol.value(costJ)
    optTraj.costTime = sol.value(costTime)
    optTraj.costWork = sol.value(costWork)
    optTraj.costFR = sol.value(costFR)
    optTraj.uraterate = sol.value(uraterate)
    optTraj.duration = sol.value(duration)

    ### compute peak handspeed and peak speed
    handspeed_opt = np.ndarray([optTraj.Q.shape[1]])
    for i in range(0,optTraj.U.shape[1]):
      qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
      qdottemp = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
      handspeed_opt[i],dum = self.handspeed(qtemp,qdottemp)
    optTraj.handspeed = handspeed_opt
    peakhandspeed = max(handspeed_opt)
    optTraj.peakhandspeed = peakhandspeed
    ### /compute peak handspeed and peak speed

    if theGeneratePlots:
      optTraj.generatePlots()

    return optTraj.duration, optTraj.costJ, optTraj.costWork, optTraj.costFR, optTraj.costTime, optTraj.peakhandspeed, optTraj  