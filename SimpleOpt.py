# SimpleModel structure
#%%
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.interpolate 

# animation 
from matplotlib import animation as animate
from IPython import display


class optTrajectories:
  mechPower = np.array([0])
  Q         = np.array([0])
  QDot      = np.array([0])
  U         = np.array([0])
  time      = np.array([0])
  handspeed = np.array([0])
  uraterate = np.array([0])
  hand      = np.array([0])

  solved        = 0
  costJ         = 0
  costFR        = 0
  costWork      = 0
  costTime      = 0
  duration      = 0
  peakhandspeed = 0
  
  def __init__(self,solved = True):
    self.solved = solved

  def generatePlots(self):
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.Q[0,:],
                            self.time, self.Q[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('generalized coordinates [° or m]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.hand[0,:],
                            self.hand[1,:])
      ax.axis('equal')
      plt.xlabel('hand x position [m]')
      plt.ylabel('hand y position [m]')
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.QDot[0,:],
                            self.time, self.QDot[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('angular velocity [s^-1]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      # Joint torques.
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.U[0,:],
                            self.time, self.U[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('Joint torques [Nm]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.uraterate[0,:],
                            self.time, self.uraterate[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('force rate rate [N/s^2]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.handspeed)
      plt.xlabel('Time [s]')
      plt.ylabel('Hand speed [m/s]')
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.mechPower[0,:], 
                            self.time, self.mechPower[1,:], 
                            )
      plt.xlabel('Time [s]')
      plt.ylabel('Power [W]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

class optiParam:
  opti = []           # the casadi optimization class.
  N = []              # number of nodes.

  qstart = []        # starting states
  qend = []           # ending states
  
  Q = []              # all states, capital Q.
  q = []              # generalized states
  dqdt = []           # generalized velocities
  ddqdt2 = []         # generalized accelerations
  u = []              # generalized torques
  kFR =[]          # force rate coefficient
  kWork = []       # work coefficient, sometimes 4.2 {margaria}.
  dudt = []           # dfr dt
  ddudt2 = []         # ddfr dt2
  sol = []            # previous solution. JDW NOTE : Check this.
  costJ = []          # objective value, typically J = Energy(work,costFR*frr) + timeValuation*Duration
  costWork = []       # cost Work
  costFR = []         # cost FR
  costTime = []       # cost Time
  timeValuation = []  # coefficient to convert time to Joules.
  parameter_names = []# unused
  duration = []       # movement duration.
  mechPower = []      # mechanical power
  pPower = []         # positive power
  time = []           # time vector.
  discreteOrCont = [] # yes. is the movement discrete endpoint to endpoint, or cyclic? 

  def __init__(self,optiIn:ca.Opti, N:int):
    self.opti = optiIn
    self.N = N

class SimpleModelEnergy:
  e_k = np.array([0])
  e_g = np.array([0])
  e_mech = np.array([0])
  e_mechAll = np.array([0])

# minjerk normalized trajectory across time array T. 
# input/2: T (array)
# output: smooth sigmoid from 0 to 1 across T. 
def minjerk(theN:int):
  T = np.linspace(0,1,int(theN))
  return 10*(T)**3 - 15*(T)**4 + 6*(T)**5

### OBJECTIVE 
def trapInt(t,inVec):
  sumval = 0.
  for ii in range(0,inVec.shape[1]-1):
      sumval = sumval + (inVec[ii]+inVec[ii+1])/2.0 * (t[ii+1]-t[ii])
  return sumval

def HermiteSimpsonImplicit(opti:ca.Opti, theDDTFun:callable, theEOMImplicitFun:callable, theQ, theQDotDot, theUIn,dt,indQDot=[2,3]):
  for k in range(theQ.shape[1]-1):
    # Hermite Simpson quadrature for all derivatives. 
    f = theDDTFun(theQ[:, k], theUIn[:, k], theQDotDot[:,k])
    fnext = theDDTFun(theQ[:, k+1], theUIn[:, k+1], theQDotDot[:,k])
    qhalf = 0.5*(theQ[:, k] + theQ[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(theUIn[:, k] + theUIn[:, k+1])
    acchalf = 0.5*(theQDotDot[:, k] + theQDotDot[:, k+1]) + dt/8*(f[indQDot,:] - fnext[indQDot,:])
    fhalf = theDDTFun(qhalf, uhalf, acchalf)
    opti.subject_to(theQ[:, k+1] - theQ[:, k] == dt/6 * (f + 4*fhalf + fnext))  # close the gaps
    ### CONSTRAINT: Implicit dynamics, M*A - F = 0
    opti.subject_to(theEOMImplicitFun(theQ[:,k],theQDotDot[:,k]) == 0) 

# How is this explicit? 
# In the implicit form above, we set the derivative of qdqdt to be a variable, acc.
# we then say that M*acc = sum(tau). 
def HermiteSimpson(opti:ca.Opti, theDDTFun:callable, theEOMImplicitFun:callable, theQ, theQDotDot, theUIn, dt, indQDot = [2,3]):
  for k in range(theQ.shape[1]-1):
    # Hermite Simpson quadrature for all derivatives. 
    f = theDDTFun(theQ[:, k], theUIn[:, k], theQDotDot[:,k])
    fnext = theDDTFun(theQ[:, k+1], theUIn[:, k+1], theQDotDot[:,k])
    qhalf = 0.5*(theQ[:, k] + theQ[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(theUIn[:, k] + theUIn[:, k+1])
    acchalf = 0.5*(theQDotDot[:, k] + theQDotDot[:, k+1]) + dt/8*(f[indQDot,:] - fnext[indQDot,:])
    fhalf = theDDTFun(qhalf, uhalf, acchalf)
    opti.subject_to(theQ[:, k+1] - theQ[:, k] == dt/6 * (f + 4*fhalf + fnext))  # close the gaps
    ### CONSTRAINT: Implicit dynamics, M*A - F = 0
    opti.subject_to(theEOMImplicitFun(theQ[:,k],theQDotDot[:,k]) == 0) 

class SimpleModel:
  l = np.array([0])
  d = np.array([0])
  I = np.array([0])
  m = np.array([0])
  g = 0
  DoF = 2
  NBodies = 2

  def animate(self, filename = ""):
    return 0
  def elbow(self,q):
    return 0
  ##############################################################################################################################################################################################
  #F - m*a = 0
  def implicitEOM(self, q, u, acc):
    return 0
  ##############################################################################################################################################################################################
  def implicitEOMQ(self, Q, acc):
    return 0
  ##############################################################################################################################################################################################
  def inverseDynamics(self, q, qdot, qdotdot):
    return 0
  ##############################################################################################################################################################################################
  def kinematicJacobianInertias(self,q):
    return 0
  ##############################################################################################################################################################################################
  def kinematicJacobianRotationalInertias(self):
    return 0
  ##############################################################################################################################################################################################
  def kinematicJacobianEndpoint(self, q):
    return 0
  ##############################################################################################################################################################################################
  def joints2Endpoint(self,q):
    return 0
  ##############################################################################################################################################################################################
  def joints2EndpointSymbolic(self,q):
    return 0
  ##############################################################################################################################################################################################
  def jointPower(self,q):
    return 0
  ##############################################################################################################################################################################################
  def xy2joints(self,xy):
    return 0
  ##############################################################################################################################################################################################
  def handspeed(self, q, qdot):
    return 0
  ##############################################################################################################################################################################################
  def heightsMasses(self,q):
    return [q[0]]
  ##############################################################################################################################################################################################
  def setHandMass(self,q):
    return 0
  

  # def energy(self, theQ, theQDot, theU, theT):
  # computes general energy balance for a model with kinematicJacobianInertias and kinematicJacobianRotationalInertias defined. 
  # note: this code expects that the input matrices are nQxnT. 
  def energy(self, theQ, theQDot, theU, theT):
    
    # mechanical power
    eDot_mech = self.jointPower(theQDot,theU)
    t = theT.reshape([1,-1])
    e_mechAll = integ.cumulative_trapezoid(eDot_mech,x = t,initial=0) #axis = -1 by default, the last axis [which is true], initial keeps shape same.
    e_mech = np.zeros([1,e_mechAll.shape[1]])
    #  one row for e_mech
    for iRow in range(0,e_mechAll.shape[0]):
      e_mech = e_mech + e_mechAll[iRow:iRow+1,:]
    
    # gravitational work.
    e_g = np.zeros([1,theQ.shape[1]])
    for it in np.arange(0,theQ.shape[1]):
      heights = self.heightsMasses(theQ[:,it])
      for ih in np.arange(0,len(heights)):
        e_g[0,it] = e_g[0,it] + self.m[ih]*self.g*heights[ih]
    #e_g = e_g - e_g[0] # subtract initial

    # kinetic energy
    e_k = np.zeros([1,theQ.shape[1]])
    for it in np.arange(0,theQ.shape[1]):
      linjac = self.kinematicJacobianInertias(theQ[:,it])
      rotjac = self.kinematicJacobianRotationalInertias()
      for ib in np.arange(0,self.NBodies):
        vt = linjac[ib] @ theQDot[:,it]
        angvt = rotjac[ib] @ theQDot[:,it]
        e_k[0,it] = e_k[0,it] + 1/2 * self.m[ib] * vt.T @ vt
        e_k[0,it] = e_k[0,it] + 1/2 * self.I[ib] * angvt.T @ angvt

    energyOut = SimpleModelEnergy()
    energyOut.e_g = e_g.flatten()
    energyOut.e_k = e_k.flatten()
    energyOut.e_mechAll = e_mechAll
    energyOut.e_mech = e_mech.flatten()
    return energyOut

  def guessWithTrajectory(self,op1:optiParam, op2:optiParam):
    
    return 0
  
  # def movementTimeOptSetup(self, 
  #  theN              = 100, 
  #  theFRCoef         = 8.5e-2, 
  #  theWorkCoef       = 4.2,
  #  theTimeValuation  = 1, 
  #  theDuration       = [], #if empty, we are optimizing for duration. 
  #  theDurationGuess  = .5):
  # Trajectory optimization problem formulation, hermite simpson, implicit dynamics constraints means
  # We ask the solver to find (for example) accelerations and write the equations of motion implicitly. 
  def movementTimeOptSetup(self, 
    theN              = 100, 
    theFRCoef         = 8.5e-2, 
    theWorkCoef       = 4.2,
    theTimeValuation  = 1, 
    theDuration       = [], #if empty, we are optimizing for duration. 
    theDurationGuess  = .5,
    theHandMass       = 0.0,
    discreteOrCont    = 'discrete') -> optiParam:

    #### the casadi instance of Opti helper class. 
    opti = ca.Opti()
    oP   = optiParam(opti, N = theN) # we attach all symbolic opt variables to oP, to be returned.
    
    #### STATE (Q), Acceleration (ddqdt2), force-rate-rate
    oP.Q         = opti.variable(self.DoF*4, theN+1)      # big-Q
    oP.ddqdt2    = opti.variable(self.DoF,   theN+1)      # optimal acceleration, for implicit equations of motion.
    oP.ddudt2    = opti.variable(self.DoF,   theN+1)      # Force rate
    
    #### slack variables
    slackVars = opti.variable(self.DoF*3, theN+1)   # for computing positive power, and +/- frr.
    
    #### parameters: 
    # these can change from opt to opt without re-setting up the optimization.
    oP.qstart        = opti.parameter(self.DoF,1)
    oP.qend          = opti.parameter(self.DoF,1)
    oP.timeValuation = opti.parameter()
    opti.set_value(oP.timeValuation, theTimeValuation)
    oP.kFR           = opti.parameter()
    opti.set_value(oP.kFR, theFRCoef)
    self.setHandMass(theHandMass) #this may be converted to parameter. but requires re-applying dynamics constraints.
    ###/ END State, acceleration (implicit method), force-rate, slack vars.
    
    #### Define movement duration as either optimized, or fixed. 
    # Then this code can be used for both types of optimizations. 
    if not(theDuration):                        ### FIRST: optimized param: make it an opti.variable()
      oP.duration = opti.variable() 
      opti.subject_to(oP.duration >   0.0)           # critical!
      opti.subject_to(oP.duration <= 20.0)          # maybe unnecessary! =)
      opti.set_initial(oP.duration,theDurationGuess)
      oP.time = ca.linspace(0., oP.duration, theN+1)  # Discretized time vector
      opti.subject_to(oP.time[:] >= 0.) 
    else:                                       ### SECOND: fixed duration.
      oP.duration = theDuration
      oP.time     = ca.linspace(0., oP.duration, theN+1)  # Discretized time vector
    dt = (oP.duration)/theN

    # extract columns of Q for handiness.
    oP.q     = oP.Q[self.DoF*0 : self.DoF*1, :] # position    
    oP.dqdt  = oP.Q[self.DoF*1 : self.DoF*2, :] # velocity
    oP.u     = oP.Q[self.DoF*2 : self.DoF*3, :] # force
    oP.dudt  = oP.Q[self.DoF*3 : self.DoF*4, :] # force rate
    
    # Calculus equation constraint
    def qd(qi, ui, acc): 
      return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6], qi[7], ui[0], ui[1])  # dq/dt = f(q,u)
    
    #### IMPLEMENT DYNAMICS CONSTRAINTS. in subfunction, for now only HermiteSimpsonImplicit.
    HermiteSimpsonImplicit(opti, qd, self.implicitEOMQ, oP.Q, oP.ddqdt2, oP.ddudt2, dt, [2,3])
    
    #### DEFINE SLACK VARIABLES: Primarily for power constraints, also for force rate. 
    # extract slack variables for power
    oP.pPower = slackVars[0:self.DoF,      :]
    pddu = slackVars[self.DoF:self.DoF*2,  :]
    nddu = slackVars[self.DoF*2:self.DoF*3,:]

    oP.mechPower = self.jointPower(oP.dqdt,oP.u)
    # Constrain positive power variable 'pPower' 
    opti.subject_to(oP.pPower[0,:] >= 0.) 
    opti.subject_to(oP.pPower[1,:] >= 0.) 
    opti.subject_to(oP.pPower[0,:] >= oP.mechPower[0,:]) 
    opti.subject_to(oP.pPower[1,:] >= oP.mechPower[1,:]) 
    # Constrain positive and negative fraterate 
    opti.subject_to(pddu[0,:] >= 0.)  
    opti.subject_to(pddu[1,:] >= 0.)  
    opti.subject_to(pddu[0,:] >= oP.ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= oP.ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= oP.ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= oP.ddudt2[1,:]) 
    
    #### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY CONSTRAINTS) ####
    # Boundary constraints. Often zeros
    def initAndEndZeros(opti:ca.Opti,thelist):
      for var in thelist:
        for dof in range(0,var.shape[0]):
          opti.subject_to(var[dof, 0] == 0.0)
          opti.subject_to(var[dof,-1] == 0.0)
      
    def initAndEndMatch(opti:ca.Opti, thelist):
        for var in thelist:
          for dof in range(0, var.shape[0]):
            opti.subject_to(var[dof,0] == var[dof,-1])
    
    # discrete or continuous
    oP.discreteOrCont = discreteOrCont
    if discreteOrCont == 'continuous':
      initAndEndZeros(opti,[oP.dqdt])
      initAndEndMatch(opti,[oP.u, oP.ddudt2, pddu, nddu])  
      opti.subject_to(oP.q[:,0]           == oP.qstart)
      opti.subject_to(oP.q[:,-1]          == oP.qstart)
      opti.subject_to(oP.q[:,theN/2]      == oP.qend)
      opti.subject_to(oP.dqdt[:,theN/2]   == 0)

    else:
      initAndEndZeros(opti,[oP.dqdt,oP.ddudt2,oP.ddqdt2,pddu,nddu]) # be careful here.
      opti.subject_to(oP.ddqdt2[:,-2] == 0)                         # 2022-11. here we are accounting for hermite-simpson using fwd-estimates. 
      opti.subject_to(oP.q[:,0]   == oP.qstart)
      opti.subject_to(oP.q[:,-1]  == oP.qend)
    
    #### OBJECTIVE ##### 
    # this cost is per movement. We intergrate across time to get joules. 
    oP.costTime  = oP.time[-1] * oP.timeValuation
    oP.kWork     = theWorkCoef
    oP.costWork  = oP.kWork * (trapInt(oP.time,oP.pPower[0,:]) +\
                               trapInt(oP.time,oP.pPower[1,:]))
    oP.costFR    = oP.kFR * (trapInt(oP.time, pddu[0,:]) +\
                               trapInt(oP.time, pddu[1,:]) -\
                               trapInt(oP.time, nddu[0,:]) -\
                               trapInt(oP.time, nddu[1,:]))
    oP.costJ     = oP.costTime + oP.costWork + oP.costFR
    # Set cost function
    opti.minimize(oP.costJ)

    #### Hyperparameters and plotting function #### 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter"        : maxIter,
            "constr_viol_tol" : 1e-2,
            "dual_inf_tol"    : 1e-2}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(oP.time),opti.debug.value(oP.q[0,:]),
          opti.debug.value(oP.time), opti.debug.value(oP.q[1,:]),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)

    return oP

  def updateGuessAndSolve(self,
    oP:optiParam,
    xystart:np.ndarray,
    xyend:np.ndarray,
    theDurationGuess      = 1.0,
    theTimeValuation      = 1.0,
    theGeneratePlots      = 1,
    theFRCoef             = 8.5e-2):
    
    # unpack the opti variables for ease of use. 
    theN    = oP.N                       # number of nodes
    opti    = oP.opti                    # opti optimization framework
    qCON0   = self.xy2joints(xystart) # q starting
    qCON1   = self.xy2joints(xyend)   # q ending
    if (sum(np.isnan(qCON0))+sum(np.isnan(qCON1)))>0:
      print("error: cannot reach this target.")
      return optTrajectories(solved = False), oP
    q       = oP.q
    dqdt    = oP.dqdt
    ddqdt2  = oP.ddqdt2
    u       = oP.u
    dudt    = oP.dudt
    ddudt2  = oP.ddudt2

    # update the parameters of the optimization
    opti.set_value(oP.qstart,         qCON0)      
    opti.set_value(oP.qend,           qCON1)
    opti.set_value(oP.timeValuation,  theTimeValuation)
    opti.set_value(oP.kFR,            theFRCoef)

    # now update the time for guesses of the decision variables.
    if type(oP.duration) == float:
      theDurationGuess = oP.duration
      print("Leaving duration at setup: " + str(oP.duration) + " s.")
    else:
      opti.set_initial(oP.duration, theDurationGuess)  
    tGuess = np.linspace(0,theDurationGuess,theN+1)

    # form guesses. 
    mj = minjerk(theN+1)
    if oP.discreteOrCont == 'continuous':
      mj1 = minjerk(int(theN/2))
      mj2 = minjerk(int(theN/2+1))
      mj = np.concatenate([mj1,mj2[::-1]])

    nQ = q.shape[0]
    nT = q.shape[1]
    qGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      qGuess[qloop,:] = qCON0[qloop] + mj*(qCON1[qloop]-qCON0[qloop])
    opti.set_initial(q, qGuess)
    
    dqdtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dqdtGuess[qloop,:] = np.gradient(qGuess[qloop,:],tGuess)
    opti.set_initial(dqdt, dqdtGuess)

    ddqdt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dqdtGuess[qloop,:],tGuess)
    opti.set_initial(ddqdt2, ddqdt2Guess)

    uGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      for tloop in range(0,nT):
        uGuess[:,tloop:tloop+1] = self.inverseDynamics(qGuess[:,tloop],dqdtGuess[:,tloop],ddqdt2Guess[:,tloop])
    opti.set_initial(u, uGuess)
    
    # slightly decreases solver iterations to provide a guess.
    dudtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dudtGuess[qloop,:] = np.gradient(uGuess[qloop,:],tGuess)
    opti.set_initial(dudt, dudtGuess)

    ddudt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddudt2Guess[qloop,:] = np.gradient(dudtGuess[qloop,:],tGuess)
    opti.set_initial(ddudt2, ddudt2Guess)

    try:
      sol = opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved = True)
      optTraj.time      = sol.value(oP.time)
      optTraj.Q         = sol.value(q)
      optTraj.QDot      = sol.value(dqdt)
      optTraj.U         = sol.value(u)
      optTraj.mechPower = sol.value(oP.mechPower)
      optTraj.costJ     = sol.value(oP.costJ)
      optTraj.costTime  = sol.value(oP.costTime)
      optTraj.costWork  = sol.value(oP.costWork)
      optTraj.costFR    = sol.value(oP.costFR)
      optTraj.uraterate = sol.value(oP.ddudt2)
      optTraj.duration  = sol.value(oP.duration)

      ### compute peak handspeed and peak speed
      handspeed_opt = np.zeros([optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        qdottemp = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
        handspeed_opt[i],dum = self.handspeed(qtemp,qdottemp)
      
      optTraj.handspeed = handspeed_opt
      peakhandspeed = max(handspeed_opt)
      optTraj.peakhandspeed = peakhandspeed
      ### /compute peak handspeed and peak speed
      
      hand_opt = np.zeros([2,optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        hand_opt[:,i] = self.joints2Endpoint(qtemp)
      optTraj.hand = hand_opt
      

      # plot
      if theGeneratePlots:
        optTraj.generatePlots()

      #return solution
      oP.opti = opti
      return optTraj, oP
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = opti.debug.value(oP.time)
      failTraj.Q         = opti.debug.value(q)
      failTraj.QDot      = opti.debug.value(dqdt)
      failTraj.U         = opti.debug.value(u)
      failTraj.mechPower = opti.debug.value(oP.mechPower)
      failTraj.costJ     = opti.debug.value(oP.costJ)
      failTraj.costTime  = opti.debug.value(oP.costTime)
      failTraj.costWork  = opti.debug.value(oP.costWork)
      failTraj.costFR    = opti.debug.value(oP.costFR)
      failTraj.uraterate = opti.debug.value(oP.ddudt2)
      failTraj.duration  = opti.debug.value(oP.duration)
      
      return failTraj, oP

  def constrainShoulder(self,oP:optiParam):
    dqdt = optiParam.dqdt
    ddqdt2 = optiParam.ddqdt2
    for i in range(0,dqdt.shape[2]):
      oP.opti.subject_to(dqdt[0,i] == 0)
      oP.opti.subject_to(ddqdt2[0,i] == 0)

  def interpolateGuessAndSolve(self,
    oPLow:optiParam,
    oPHigh:optiParam,
    theGeneratePlots = True):
    
    # unpack the opti variables for ease of use. 
    theHighN    = oPHigh.N                       # number of nodes
    

    tLow    = oPLow.opti.value(oPLow.time)
    tGuess  = np.linspace(0,tLow[-1],theHighN+1)
     
    #### update the parameters of the optimization:
    # endpoint constraints
    # force rate.
    # time valuation. 
    # JDW: should consider adding kWork.
    oPHigh.opti.set_value(oPHigh.qstart,         oPLow.opti.value(oPLow.qstart))      
    oPHigh.opti.set_value(oPHigh.qend,           oPLow.opti.value(oPLow.qend))
    oPHigh.opti.set_value(oPHigh.timeValuation,  oPLow.opti.value(oPLow.timeValuation))
    oPHigh.opti.set_value(oPHigh.kFR,            oPLow.opti.value(oPLow.costFR))


    #### interpolate
    mj = minjerk(theHighN+1)
    if oPLow.discreteOrCont == 'continuous':
      mj1 = minjerk(int(theHighN/2))
      mj2 = minjerk(int(theHighN/2+1))
      mj = np.concatenate([mj1,mj2[::-1]])

    nQ = oPHigh.q.shape[0]
    nT = oPHigh.q.shape[1]

    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.q.shape[0]):
      lowVal                = oPLow.opti.value(oPLow.q)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.q, tempGuess)
    
    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.dqdt.shape[0]):
      lowVal              = oPLow.opti.value(oPLow.dqdt)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.dqdt, tempGuess)

    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.ddqdt2.shape[0]):
      lowVal              = oPLow.opti.value(oPLow.ddqdt2)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.ddqdt2, tempGuess)

    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.u.shape[0]):
      lowVal              = oPLow.opti.value(oPLow.u)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.u, tempGuess)
    
    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.dudt.shape[0]):
      lowVal              = oPLow.opti.value(oPLow.dudt)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.dudt, tempGuess)

    tempGuess = np.zeros([nQ,nT])
    for qloop in range(0,oPHigh.ddudt2.shape[0]):
      lowVal              = oPLow.opti.value(oPLow.ddudt2)
      modelTemp           = scipy.interpolate.splrep(tLow, lowVal[qloop,:], s=3) #build a spline representation of C. s=0 means no smoothing
      tempGuess[qloop,:]  = scipy.interpolate.splev(tGuess,modelTemp)
    oPHigh.opti.set_initial(oPHigh.ddudt2, tempGuess)

    try:
      sol = oPHigh.opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved = True)
      optTraj.time      = sol.value(oPHigh.time)
      optTraj.Q         = sol.value(oPHigh.q)
      optTraj.QDot      = sol.value(oPHigh.dqdt)
      optTraj.U         = sol.value(oPHigh.u)
      optTraj.mechPower = sol.value(oPHigh.mechPower)
      optTraj.costJ     = sol.value(oPHigh.costJ)
      optTraj.costTime  = sol.value(oPHigh.costTime)
      optTraj.costWork  = sol.value(oPHigh.costWork)
      optTraj.costFR    = sol.value(oPHigh.costFR)
      optTraj.uraterate = sol.value(oPHigh.ddudt2)
      optTraj.duration  = sol.value(oPHigh.duration)

      ### compute peak handspeed and peak speed
      handspeed_opt = np.zeros([optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp                 = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        qdottemp              = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
        handspeed_opt[i],dum  = self.handspeed(qtemp,qdottemp)
      
      optTraj.handspeed     = handspeed_opt
      optTraj.peakhandspeed = max(handspeed_opt)
      ### /compute peak handspeed and peak speed
      
      hand_opt = np.zeros([2,optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        hand_opt[:,i:i+1] = self.joints2Endpoint(qtemp)
      optTraj.hand = hand_opt
      

      # plot
      if theGeneratePlots:
        optTraj.generatePlots()

      #return solution
      return optTraj, oPHigh
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = oPHigh.opti.debug.value(oPHigh.time)
      failTraj.Q         = oPHigh.opti.debug.value(oPHigh.q)
      failTraj.QDot      = oPHigh.opti.debug.value(oPHigh.dqdt)
      failTraj.U         = oPHigh.opti.debug.value(oPHigh.u)
      failTraj.mechPower = oPHigh.opti.debug.value(oPHigh.mechPower)
      failTraj.costJ     = oPHigh.opti.debug.value(oPHigh.costJ)
      failTraj.costTime  = oPHigh.opti.debug.value(oPHigh.costTime)
      failTraj.costWork  = oPHigh.opti.debug.value(oPHigh.costWork)
      failTraj.costFR    = oPHigh.opti.debug.value(oPHigh.costFR)
      failTraj.uraterate = oPHigh.opti.debug.value(oPHigh.ddudt2)
      failTraj.duration  = oPHigh.opti.debug.value(oPHigh.duration)
      
      return failTraj, oPHigh


  def solvewithWarmTraj(self,oP:optiParam, xstartnew:np.ndarray, xendnew:np.ndarray, warmTraj:optTrajectories, \
    theTimeValuation  = 1.0, \
    theGeneratePlots  = 1,
    theFRCoef = 8.5e-2):
    
    # unpack the opti variables for ease of use. 
    theN    = oP.N                       # number of nodes
    opti    = oP.opti                    # opti optimization framework
    
    # get q for xyhand; return if imaginary (nan)
    qCON0   = self.xy2joints(xstartnew) # q starting
    qCON1   = self.xy2joints(xendnew)   # q ending
    if (sum(np.isnan(qCON0))+sum(np.isnan(qCON1)))>0:
      print("error: cannot reach this target.")
      return optTrajectories(solved = False), oP
    
    # simplified access to some casadi variables stored in oP.
    q       = oP.q
    dqdt    = oP.dqdt
    ddqdt2  = oP.ddqdt2
    u       = oP.u

    # update the parameters of the optimization
    opti.set_value(oP.qstart,         qCON0)      
    opti.set_value(oP.qend,           qCON1)
    opti.set_value(oP.timeValuation,  theTimeValuation)
    opti.set_value(oP.frCoef, theFRCoef)

    # update the duration guess
    opti.set_initial(oP.duration, warmTraj.time[-1])
    tGuess = np.linspace(0,       warmTraj.time[-1],  theN+1)
    mj = minjerk(theN)

    # what do we want to do? let's use the 
    nQ = q.shape[0]
    nT = q.shape[1]
    qGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      qGuess[qloop,:] = qCON0[qloop] + mj*(qCON1[qloop]-qCON0[qloop])
    opti.set_initial(q, qGuess)
    
    dqdtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dqdtGuess[qloop,:] = np.gradient(qGuess[qloop,:],tGuess)
    opti.set_initial(dqdt, dqdtGuess)

    ddqdt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dqdtGuess[qloop,:],tGuess)
    opti.set_initial(ddqdt2, ddqdt2Guess)

    uGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      for tloop in range(0,nT):
        uGuess[:,tloop:tloop+1] = self.inverseDynamics(qGuess[:,tloop],dqdtGuess[:,tloop],ddqdt2Guess[:,tloop])
    opti.set_initial(u, uGuess)

    try:
      sol = opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved = True)
      optTraj.time      = sol.value(oP.time)
      optTraj.Q         = sol.value(q)
      optTraj.QDot      = sol.value(dqdt)
      optTraj.U         = sol.value(u)
      optTraj.mechPower = sol.value(oP.mechPower)
      optTraj.costJ     = sol.value(oP.costJ)
      optTraj.costTime  = sol.value(oP.costTime)
      optTraj.costWork  = sol.value(oP.costWork)
      optTraj.costFR    = sol.value(oP.costFR)
      optTraj.uraterate = sol.value(oP.ddudt2)
      optTraj.duration  = sol.value(oP.duration)

      ### compute peak handspeed and peak speed
      handspeed_opt = np.zeros([optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        qdottemp = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
        handspeed_opt[i],dum = self.handspeed(qtemp,qdottemp)
      
      optTraj.handspeed = handspeed_opt
      peakhandspeed = max(handspeed_opt)
      optTraj.peakhandspeed = peakhandspeed
      ### /compute peak handspeed and peak speed

      # plot
      if theGeneratePlots:
        optTraj.generatePlots()

      #return solution
      oP.opti = opti
      return optTraj, oP
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = opti.debug.value(oP.time)
      failTraj.Q         = opti.debug.value(q)
      failTraj.QDot      = opti.debug.value(dqdt)
      failTraj.U         = opti.debug.value(u)
      failTraj.mechPower = opti.debug.value(oP.mechPower)
      failTraj.costJ     = opti.debug.value(oP.costJ)
      failTraj.costTime  = opti.debug.value(oP.costTime)
      failTraj.costWork  = opti.debug.value(oP.costWork)
      failTraj.costFR    = opti.debug.value(oP.costFR)
      failTraj.uraterate = opti.debug.value(oP.ddudt2)
      failTraj.duration  = opti.debug.value(oP.duration)
      oP.copy
      return failTraj, oP

  def movementTimeNoXSetup(self, 
    theN              = 100, 
    theFRCoef         = 8.5e-2, 
    theTimeValuation  = 1, 
    theDuration       = [], #if empty, we are optimizing for duration. 
    theDurationGuess  = .5,
    gt1lt0            = 1,
    theKWork          = 4.2,
    theDiscreteOrCont = 'discrete'):

    opti = ca.Opti()
    # all opti variables will be attached to the oP instance of optiParam
    oP = optiParam(opti,N = theN)

    ### Define STATE (Q), Acceleration (ddqdt2), force-rate, SLACKVARS, 
    ### and Parameters: qstart, qend, timeValuation, frCoef 
    oP.N = theN
    oP.Q             = opti.variable(self.DoF*4, theN+1)
    # optimal acceleration, for implicit equations of motion.
    oP.ddqdt2        = opti.variable(self.DoF,theN+1)
    # Force rate
    oP.ddudt2        = opti.variable(self.DoF, theN+1)
    # slack variables
    slackVars        = opti.variable(self.DoF*3, theN+1)  #fully-actuated, constrain 1:fr_p, 2:fr_n, 3:power_p.
    
    # parameters: these can change from opt to opt
    oP.qstart        = opti.parameter(self.DoF,1)
    oP.yBoundary     = opti.parameter(1,1)
    oP.timeValuation = opti.parameter()
    opti.set_value(oP.timeValuation, theTimeValuation)
    oP.kFR           = opti.parameter()
    oP.kWork         = theKWork
    opti.set_value(oP.kFR, theFRCoef)
    ###/
    
    ### Define movement duration as either optimized, or fixed. 
    ### if we are optimizing movement time, make it an opti.variable()
    ###, and solve for it. place some loose bounds on duration.
    if not(theDuration):
      oP.duration = opti.variable() 
      opti.subject_to(oP.duration > 0.0)  # critical!
      opti.subject_to(oP.duration <=20.0) # maybe unnecessary! =)
      
      durationInitial = theDurationGuess
      opti.set_initial(oP.duration,durationInitial)
    else:
      oP.duration     = theDuration
      durationInitial = oP.duration 
    dt = (oP.duration)/theN
    oP.time = ca.linspace(0., oP.duration, theN+1)  # Discretized time vector
    ###/

    ### extract columns of Q for handiness.
    # position
    oP.q    = oP.Q[0:2,:]
    q1      = oP.Q[0, :]
    q2      = oP.Q[1, :]
    # velocity
    oP.dqdt = oP.Q[2:4,:]
    # force
    oP.u    = oP.Q[4:6,:]
    # force rate
    oP.dudt = oP.Q[6:8,:]
    # /extraction

    ### Calculus equation constraint
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete nodes and enforce calculus constraints. 
    HermiteSimpsonImplicit(opti,qd,self.implicitEOMQ,oP.Q,oP.ddqdt2,oP.ddudt2,dt,[2,3])
    
    # CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
    # variables will be bounded between +/- Inf).
    # for i in range(0,q.shape[0]):
    #   opti.subject_to(opti.bounded(-2*ca.pi, q[i,:], 2*ca.pi))

    ##### SLACK VARIABLES: Primarily for power constraints, also for force rate. 
    ### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
    # # extract slack variables for power and force-rate-rate for handiness.
    pPower = slackVars[0:self.DoF,         :]
    pddu = slackVars[self.DoF:self.DoF*2,  :]
    nddu = slackVars[self.DoF*2:self.DoF*3,:]

    oP.mechPower = self.jointPower(oP.dqdt, oP.u)
    # positive power constraints
    opti.subject_to(oP.time[:] >= 0.) 
    opti.subject_to(pPower[0,:] >= 0.) 
    opti.subject_to(pPower[1,:] >= 0.) 
    opti.subject_to(pPower[0,:] >= oP.mechPower[0,:]) 
    opti.subject_to(pPower[1,:] >= oP.mechPower[1,:]) 
    # fraterate constraints
    opti.subject_to(pddu[0,:] >= 0.)  
    opti.subject_to(pddu[1,:] >= 0.)  
    opti.subject_to(pddu[0,:] >= oP.ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= oP.ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= oP.ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= oP.ddudt2[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY CONSTRAINTS) ####################################
    # Boundary constraints. Often zeros
    def initAndEndZeros(opti,list):
      for var in list:
        for dof in range(0,var.shape[0]):
          opti.subject_to(var[dof,0] == 0.0)
          opti.subject_to(var[dof,-1] == 0.0)
    
    def initAndEndMatch(opti:ca.Opti, thelist):
        for var in thelist:
          for dof in range(0, var.shape[0]):
            opti.subject_to(var[dof,0] == var[dof,-1])
  
    # discrete or continuous
    oP.discreteOrCont = theDiscreteOrCont
    if oP.discreteOrCont == 'continuous':
      initAndEndZeros(opti,[oP.dqdt])                                     #this may be dubious. 
      initAndEndMatch(opti,[oP.u, oP.ddudt2, pddu, nddu])  
      opti.subject_to(oP.q[:,0]           == oP.qstart)
      opti.subject_to(oP.q[:,-1]          == oP.qstart)
      opti.subject_to(oP.dqdt[:,theN/2]   == 0)                           # this may be dubious.

      # continuous constraints, specific to noX start.                    
      xyEnd = self.joints2EndpointSymbolic(oP.q[:,theN/2])
      if gt1lt0:
        opti.subject_to(xyEnd[1] >= oP.yBoundary)
      else:
        print("set the boundary such that yEnd must be less than the boundary.")
        opti.subject_to(xyEnd[1] <= oP.yBoundary)

    else:
      initAndEndZeros(opti,[oP.dqdt, oP.ddudt2, oP.ddqdt2, pddu, nddu])   # be careful here.
      opti.subject_to(oP.ddqdt2[:,-2] == 0)                               # 2022-11. here we are accounting for hermite-simpson using fwd-estimates. 
      opti.subject_to(oP.q[:,0]       == oP.qstart)
      xyEnd = self.joints2EndpointSymbolic(oP.q[:,theN])
      if gt1lt0:
        opti.subject_to(xyEnd[1] >= oP.yBoundary)
      else:
        print("set the boundary such that yEnd must be less than the boundary.")
        opti.subject_to(xyEnd[1] <= oP.yBoundary)
    

    ############################################################################################################################################
    ############## OBJECTIVE ############## 
    oP.costTime = oP.time[-1] * oP.timeValuation
    oP.costWork = oP.kWork * trapInt(oP.time, pPower[0,:])+trapInt(oP.time, pPower[1,:])
    oP.costFR = oP.kFR * (trapInt(oP.time,pddu[0,:]) + trapInt(oP.time,pddu[1,:]) - trapInt(oP.time,nddu[0,:]) - trapInt(oP.time,nddu[1,:]))
    oP.costJ = oP.costTime + oP.costWork + oP.costFR
    # Set cost function
    opti.minimize(oP.costJ)

    ############################################################################################################################################
    ############## Hyperparameters and solve ############## 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(oP.time),opti.debug.value(q1),
          opti.debug.value(oP.time), opti.debug.value(q2),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)

    return oP
##############################################################################################################################################################################################

  def updateNoXAndSolve(self,oP:optiParam, xstartnew:np.ndarray, theYBoundary, \
    theDurationInitial   = 1.0, \
    theTimeValuation  = 1.0, \
    theGeneratePlots  = 1,
    theFRCoef = 8.5e-2):
    
    # unpack the opti variables for ease of use. 
    theN    = oP.N                       # number of nodes
    opti    = oP.opti                    # opti optimization framework
    qCON0   = self.xy2joints(xstartnew) # q starting

    q       = oP.q
    dqdt    = oP.dqdt
    ddqdt2  = oP.ddqdt2
    u       = oP.u
    dudt    = oP.dudt
    ddudt2  = oP.ddudt2

    # update the parameters of the optimization
    opti.set_value(oP.qstart,         qCON0)      
    opti.set_value(oP.yBoundary,      theYBoundary)
    opti.set_value(oP.timeValuation,  theTimeValuation)
    opti.set_value(oP.kFR,            theFRCoef)

    # now update the guess
    opti.set_initial(oP.duration, theDurationInitial)
    tGuess = np.linspace(0,       theDurationInitial,  theN+1)
    mj = minjerk(theN+1)

    # for the sake of an initial guess, create qCON1
    qCON1 = self.xy2joints(np.array([xstartnew[0],theYBoundary]))

    nQ = q.shape[0]
    nT = q.shape[1]
    qGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      qGuess[qloop,:] = qCON0[qloop] + mj*(qCON1[qloop]-qCON0[qloop])
    opti.set_initial(q, qGuess)
    
    dqdtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dqdtGuess[qloop,:] = np.gradient(qGuess[qloop,:],tGuess)
    opti.set_initial(dqdt, dqdtGuess)

    ddqdt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dqdtGuess[qloop,:],tGuess)
    opti.set_initial(ddqdt2, ddqdt2Guess)

    uGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      for tloop in range(0,nT):
        uGuess[:,tloop:tloop+1] = self.inverseDynamics(qGuess[:,tloop],dqdtGuess[:,tloop],ddqdt2Guess[:,tloop])
    opti.set_initial(u, uGuess)

    dudtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dudtGuess[qloop,:] = np.gradient(uGuess[qloop,:],tGuess)
    opti.set_initial(dudt, dudtGuess)

    ddudt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dudtGuess[qloop,:],tGuess)
    opti.set_initial(ddudt2, ddqdt2Guess)

    try:
      sol = opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved = True)
      optTraj.time      = sol.value(oP.time)
      optTraj.Q         = sol.value(q)
      optTraj.QDot      = sol.value(dqdt)
      optTraj.U         = sol.value(u)
      optTraj.mechPower = sol.value(oP.mechPower)
      optTraj.costJ     = sol.value(oP.costJ)
      optTraj.costTime  = sol.value(oP.costTime)
      optTraj.costWork  = sol.value(oP.costWork)
      optTraj.costFR    = sol.value(oP.costFR)
      optTraj.uraterate = sol.value(oP.ddudt2)
      optTraj.duration  = sol.value(oP.duration)

      ### compute peak handspeed and peak speed
      handspeed_opt = np.zeros([optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        qdottemp = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
        handspeed_opt[i],dum = self.handspeed(qtemp,qdottemp)
      
      optTraj.handspeed = handspeed_opt
      peakhandspeed = max(handspeed_opt)
      optTraj.peakhandspeed = peakhandspeed
      ### /compute peak handspeed and peak speed

      hand_opt = np.zeros([2,optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        hand_opt[:,i] = self.joints2Endpoint(qtemp)
      optTraj.hand = hand_opt


      # plot
      if theGeneratePlots:
        optTraj.generatePlots()

      #return solution
      oP.opti = opti
      return optTraj, oP
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = opti.debug.value(oP.time)
      failTraj.Q         = opti.debug.value(q)
      failTraj.QDot      = opti.debug.value(dqdt)
      failTraj.U         = opti.debug.value(u)
      failTraj.mechPower = opti.debug.value(oP.mechPower)
      failTraj.costJ     = opti.debug.value(oP.costJ)
      failTraj.costTime  = opti.debug.value(oP.costTime)
      failTraj.costWork  = opti.debug.value(oP.costWork)
      failTraj.costFR    = opti.debug.value(oP.costFR)
      failTraj.uraterate = opti.debug.value(oP.ddudt2)
      failTraj.duration  = opti.debug.value(oP.duration)
  
      return failTraj, oP
  
  ##############################################################################################################################################################################################
  def movementSetup(self, 
    theN              = 100, 
    theFRCoef         = 8.5e-2, 
    theTimeValuation  = 1, 
    theDuration       = [], #if empty, we are optimizing for duration. 
    theDurationGuess  = .5):

    opti = ca.Opti()
    # all opti variables will be attached to the oP instance of optiParam
    oP = optiParam(opti,theN)

    ### Define STATE (Q), Acceleration (ddqdt2), force-rate, SLACKVARS, 
    ### and Parameters: qstart, qend, timeValuation, frCoef 
    oP.N = theN
    oP.Q             = opti.variable(self.DoF*4, theN+1)
    # optimal acceleration, for implicit equations of motion.
    oP.ddqdt2        = opti.variable(self.DoF,theN+1)
    # Force rate
    oP.ddudt2        = opti.variable(self.DoF, theN+1)
    # slack variables
    slackVars        = opti.variable(self.DoF*3, theN+1)  #fully-actuated, constrain 1:fr_p, 2:fr_n, 3:power_p.
    # parameters: these can change from opt to opt
    oP.qstart        = opti.parameter(self.DoF,1)
    oP.timeValuation = opti.parameter()
    opti.set_value(oP.timeValuation, theTimeValuation)
    oP.frCoef        = opti.parameter()
    opti.set_value(oP.frCoef, theFRCoef)
    ###/
    
    ### Define movement duration as either optimized, or fixed. 
    ### if we are optimizing movement time, make it an opti.variable()
    ###, and solve for it. place some loose bounds on duration.
    if not(theDuration):
      oP.duration = opti.variable() 
      opti.subject_to(oP.duration > 0.0)  # critical!
      opti.subject_to(oP.duration <=20.0) # maybe unnecessary! =)
      
      durationInitial = theDurationGuess
      opti.set_initial(oP.duration,durationInitial)
      oP.time = ca.linspace(0., oP.duration, theN+1)  # Discretized time vector
      opti.subject_to(oP.time[:] >= 0.) 
    else:
      oP.duration     = theDuration
      durationInitial = oP.duration 
      oP.time = ca.linspace(0., oP.duration)
    dt = (oP.duration) / theN
    
    
    ###/

    # extract columns of Q for handiness.
    # position
    oP.q    = oP.Q[0:2,:]
    q1      = oP.Q[0, :]
    q2      = oP.Q[1, :]
    # velocity
    oP.dqdt = oP.Q[2:4,:]
    # force
    oP.u    = oP.Q[4:6,:]
    # force rate
    oP.dudt = oP.Q[6:8,:]
    # /extraction

    # Calculus equation constraint
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete nodes and enforce calculus constraints. 
    HermiteSimpsonImplicit(opti,qd,self.implicitEOMQ,oP.Q,oP.ddqdt2,oP.ddudt2,dt,[2,3])
    
    # CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
    # variables will be bounded between +/- Inf).
    # for i in range(0,q.shape[0]):
    #   opti.subject_to(opti.bounded(-2*ca.pi, q[i,:], 2*ca.pi))

    ##### SLACK VARIABLES: Primarily for power constraints, also for force rate. 
    ### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
    # # extract slack variables for power and force-rate-rate for handiness.
    pPower = slackVars[0:self.DoF,         :]
    pddu = slackVars[self.DoF:self.DoF*2,  :]
    nddu = slackVars[self.DoF*2:self.DoF*3,:]

    oP.mechPower = self.jointPower(oP.dqdt, oP.u)
    # positive power constraints
    opti.subject_to(pPower[0,:] >= 0.) 
    opti.subject_to(pPower[1,:] >= 0.) 
    opti.subject_to(pPower[0,:] >= oP.mechPower[0,:]) 
    opti.subject_to(pPower[1,:] >= oP.mechPower[1,:]) 
    # fraterate constraints
    opti.subject_to(pddu[0,:] >= 0.)  
    opti.subject_to(pddu[1,:] >= 0.)  
    opti.subject_to(pddu[0,:] >= oP.ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= oP.ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= oP.ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= oP.ddudt2[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY CONSTRAINTS) ####################################
    # Boundary constraints. Often zeros
    def initAndEndZeros(opti,list):
      for var in list:
        for dof in range(0,var.shape[0]):
          opti.subject_to(var[dof,0] == 0.0)
          opti.subject_to(var[dof,-1] == 0.0)
    initAndEndZeros(opti,[oP.dqdt,oP.u,oP.ddudt2,oP.ddqdt2,pddu,nddu])
    
    #######################################
    ############## OBJECTIVE ############## 
    oP.costTime = oP.time[-1] * oP.timeValuation
    oP.costWork = trapInt(oP.time, pPower[0,:])+trapInt(oP.time, pPower[1,:])
    oP.costFR = oP.frCoef * (trapInt(oP.time,pddu[0,:]) + trapInt(oP.time,pddu[1,:]) - trapInt(oP.time,nddu[0,:]) - trapInt(oP.time,nddu[1,:]))
    oP.costJ = oP.costTime + oP.costWork + oP.costFR
    # Set cost function
    opti.minimize(oP.costJ)

    #######################################
    ############## Hyperparameters and solve ############## 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(oP.time),opti.debug.value(q1),
          opti.debug.value(oP.time), opti.debug.value(q2),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)

    return oP
  
  ##############################################################################################################################################################################################
  def updateChaseAndSolve(self,oP:optiParam, xstartnew:np.ndarray, 
    yoffset=.3, yspeed = 0.0,\
    theDurationGuess   = 1.0, \
    theTimeValuation  = 1.0, \
    theGeneratePlots  = 1,
    theFRCoef = 8.5e-2):
    
    # unpack the opti variables for ease of use. 
    theN    = oP.N                       # number of nodes
    opti    = oP.opti                    # opti optimization framework
    qCON0   = self.xy2joints(xstartnew) # q starting

    q       = oP.q
    dqdt    = oP.dqdt
    ddqdt2  = oP.ddqdt2
    u       = oP.u
    dudt    = oP.dudt
    ddudt2  = oP.ddudt2

    # update the parameters of the optimization
    opti.set_value(oP.qstart,         qCON0)      
    opti.set_value(oP.timeValuation,  theTimeValuation)
    opti.set_value(oP.frCoef,         theFRCoef)

    # now update the guess
    opti.set_initial(oP.duration, theDurationGuess)
    tGuess = np.linspace(0,       theDurationGuess,  theN+1)
    mj = minjerk(theN+1)

    #start
    oP.handEnd = self.joints2Endpoint(q[:,-1])
    opti.subject_to(q[0,0] == qCON0[0])
    opti.subject_to(q[1,0] == qCON0[1])

    #end hand at end for chasing.
    oP.handEnd = self.joints2Endpoint(q[:,-1])
    opti.subject_to(oP.handEnd[0] == xstartnew[0])
    opti.subject_to(oP.handEnd[1] == yoffset+yspeed*oP.duration)

    yAtDur = yoffset+yspeed*theDurationGuess
    # for the sake of an initial guess, create qCON1
    qCON1 = self.xy2joints(np.array([xstartnew[0],yAtDur]))

    nQ = q.shape[0]
    nT = q.shape[1]
    qGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      qGuess[qloop,:] = qCON0[qloop] + mj*(qCON1[qloop]-qCON0[qloop])
    opti.set_initial(q, qGuess)
    
    dqdtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dqdtGuess[qloop,:] = np.gradient(qGuess[qloop,:],tGuess)
    opti.set_initial(dqdt, dqdtGuess)

    ddqdt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dqdtGuess[qloop,:],tGuess)
    opti.set_initial(ddqdt2, ddqdt2Guess)

    uGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      for tloop in range(0,nT):
        uGuess[:,tloop:tloop+1] = self.inverseDynamics(qGuess[:,tloop],dqdtGuess[:,tloop],ddqdt2Guess[:,tloop])
    opti.set_initial(u, uGuess)

    dudtGuess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      dudtGuess[qloop,:] = np.gradient(uGuess[qloop,:],tGuess)
    opti.set_initial(dudt, dudtGuess)

    ddudt2Guess = np.zeros([nQ,nT])
    for qloop in range(0,nQ):
      ddqdt2Guess[qloop,:] = np.gradient(dudtGuess[qloop,:],tGuess)
    opti.set_initial(ddudt2, ddqdt2Guess)

    try:
      sol = opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved = True)
      optTraj.time      = sol.value(oP.time)
      optTraj.Q         = sol.value(q)
      optTraj.QDot      = sol.value(dqdt)
      optTraj.U         = sol.value(u)
      optTraj.mechPower = sol.value(oP.mechPower)
      optTraj.costJ     = sol.value(oP.costJ)
      optTraj.costTime  = sol.value(oP.costTime)
      optTraj.costWork  = sol.value(oP.costWork)
      optTraj.costFR    = sol.value(oP.costFR)
      optTraj.uraterate = sol.value(oP.ddudt2)
      optTraj.duration  = sol.value(oP.duration)

      ### compute peak handspeed and peak speed
      handspeed_opt = np.zeros([optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        qdottemp = np.array([optTraj.QDot[0,i],optTraj.QDot[1,i]])
        handspeed_opt[i],dum = self.handspeed(qtemp,qdottemp)
      
      optTraj.handspeed = handspeed_opt
      peakhandspeed = max(handspeed_opt)
      optTraj.peakhandspeed = peakhandspeed
      ### /compute peak handspeed and peak speed

      hand_opt = np.zeros([2,optTraj.Q.shape[1]])
      for i in range(0,optTraj.U.shape[1]):
        qtemp = np.array([optTraj.Q[0,i],optTraj.Q[1,i]])
        hand_opt[:,i:i+1] = self.joints2Endpoint(qtemp)
      optTraj.hand = hand_opt

      # plot
      if theGeneratePlots:
        optTraj.generatePlots()

      #return solution
      oP.opti = opti
      return optTraj, oP
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = opti.debug.value(oP.time)
      failTraj.Q         = opti.debug.value(q)
      failTraj.QDot      = opti.debug.value(dqdt)
      failTraj.U         = opti.debug.value(u)
      failTraj.mechPower = opti.debug.value(oP.mechPower)
      failTraj.costJ     = opti.debug.value(oP.costJ)
      failTraj.costTime  = opti.debug.value(oP.costTime)
      failTraj.costWork  = opti.debug.value(oP.costWork)
      failTraj.costFR    = opti.debug.value(oP.costFR)
      failTraj.uraterate = opti.debug.value(oP.ddudt2)
      failTraj.duration  = opti.debug.value(oP.duration)
  
      return failTraj, oP