# SimpleModel structure
#%%
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ

class optiParamReturn:
  opti = []
  qend = []
  qstart = []
  opti = []
  Q = []
  q = []
  dqdt = []
  ddqdt2 = []
  u = []
  frCoef =[]
  dudt = []
  ddudt2 = []
  sol = []
  N = []
  costJ = []
  costWork = []
  costFR = []
  costTime = []
  timeValuation = []
  parameter_names = []
  duration = []
  mechPower = []
  time = []

  def __init__(self,opti:ca.Opti,Q = [],theSol = [], N = [], frCoef = [], theQStart =[], theQEnd = [],q = [], dqdt = [], ddqdt2 = [],u = [], costJ = [], costWork = [], costFR = [], costTime = [], timeValuation = [],dudt = [],ddudt2 = [],mechPower = [], duration = [], time = []):
    self.opti = opti
    self.Q = Q
    self.q = q
    self.dqdt = dqdt
    self.ddqdt2 = ddqdt2
    self.u = u
    self.dudt = dudt
    self.ddudt2 = ddudt2
    self.N = N
    self.frCoef = frCoef
    self.qstart = theQStart
    self.qend = theQEnd
    self.sol = theSol
    self.costJ = costJ
    self.costTime = costTime
    self.costWork = costWork
    self.costFR = costFR
    self.timeValuation = timeValuation
    self.parameter_names = ["qstart","qend"]
    self.mechPower = mechPower
    self.duration = duration
    self.time = time

class SimpleModelEnergy:
  e_k = np.array([0])
  e_g = np.array([0])
  e_mech = np.array([0])
  e_mechAll = np.array([0])

# minjerk normalized trajectory across time array T. 
# input: T (array)
# output: smooth sigmoid from 0 to 1 across T. 
def minjerk(T):
  t_end=T[-1]
  return (10*(T/t_end)**3 - 15*(T/t_end)**4 + 6*(T/t_end)**5)

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

class SimpleModel:
  l = np.array([0])
  d = np.array([0])
  I = np.array([0])
  m = np.array([0])
  g = 0
  DoF = 2
  NBodies = 2

  ##############################################################################################################################################################################################
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

  ##############################################################################################################################################################################################
  def movementTimeOpt(self, theXYStart, theXYEnd, theN=100, theFRCoef = 8.5e-2, theTimeValuation = 1, theGeneratePlots = 1, theDuration = [], theDurationGuess = .5, LINEAR_GUESS = False):
  # def movementTimeOpt(self, xystart, xyend, N=100, generate_plots = 1):
  # Trajectory optimization problem formulation, hermite simpson, implicit dynamics constraints means
  # We ask the solver to find (for example) accelerations and write the equations of motion implicitly. 
    opti = ca.Opti()
    Q = opti.variable(8, theN+1)
    # optimal acceleration. I add a decision variable for implicit equations of motion.
    ddqdt2 = opti.variable(2,theN+1)
    # Force rate
    ddudt2 = opti.variable(2, theN+1)
    # slack variables
    slackVars = opti.variable(6, theN+1)    
    qstart = opti.parameter(self.DoF,1)
    qend = opti.parameter(self.DoF,1)
    # fixed Q start and end. 
    qCON0 = self.xy2joints(theXYStart)
    qCON1 = self.xy2joints(theXYEnd)
    opti.set_value(qstart,qCON0)      
    opti.set_value(qend,qCON1)
    
    # whether we are fixing movement time or not, here is where we do it. 
    if not(theDuration):
      duration = opti.variable() #opti.variable()
      opti.subject_to(duration >=0.0)  # critical!
      opti.subject_to(duration <=20.0) # maybe unnecessary! =)
      durationInitial = theDurationGuess
      opti.set_initial(duration,durationInitial)
    else:
      duration = theDuration
      durationInitial = duration #sets a time vector which we use to generate an initial guess. 

    # Opt variables for duration, state, and controls.  
    dt = duration/theN
    time = ca.linspace(0., duration, theN+1)  # Discretized time vector
    
    # extract columns of Q for handiness.
    # position
    q = Q[0:2,:]
    q1 = Q[0, :]
    q2 = Q[1, :]
    # velocity
    dqdt = Q[2:4,:]
    # force
    u = Q[4:6,:]
    # force rate
    dudt = Q[6:8,:]
    
    # Calculus equation constraint
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete nodes and enforce calculus constraints. 
    HermiteSimpsonImplicit(opti,qd,self.implicitEOMQ,Q,ddqdt2,ddudt2,dt,[2,3])
    
    mechPower = self.jointPower(dqdt,u)

    # CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
    # variables will be bounded between +/- Inf).
    opti.subject_to(opti.bounded(-10, q[0,:], 10))
    opti.subject_to(opti.bounded(-10, q[1,:], 10))

    ### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
    # # extract slack variables for power and force-rate-rate for handiness.
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
    opti.subject_to(pddu[0,:] >= ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= ddudt2[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY CONSTRAINTS) ####################################
    # Boundary constraints. Often zeros
    def initAndEndZeros(opti,list):
      for var in list:
        for dof in range(0,var.shape[0]):
          opti.subject_to(var[dof,0] == 0.0)
          opti.subject_to(var[dof,-1] == 0.0)
    initAndEndZeros(opti,[dqdt,u,ddudt2,ddqdt2,pddu,nddu])
    
    opti.subject_to(q[:,0] == qstart)
    opti.subject_to(q[:,-1] == qend)

    ############################################################################################################################################
    ############## OBJECTIVE ############## 
    frCoef = theFRCoef
    timeValuation = theTimeValuation
    costTime = duration * timeValuation
    costWork = trapInt(time,pPower[0,:])+trapInt(time,pPower[1,:])
    costFR = frCoef * (trapInt(time,pddu[0,:]) + trapInt(time,pddu[1,:]) - trapInt(time,nddu[0,:]) - trapInt(time,nddu[1,:]))
    costJ = costTime + costWork + costFR
    # Set cost function
    opti.minimize(costJ)

    ############################################################################################################################################
    ############## Hyperparameters and solve ############## 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(time),opti.debug.value(q1),
          opti.debug.value(time), opti.debug.value(q2),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)
    
    ############################################################################################################################################
    ############## GUESS ############## 
    if LINEAR_GUESS:
      # do nothing fancy, initialize things just off 0. 
      q1guess = np.linspace(qstart[0], qend[0], theN+1)
      q2guess = np.linspace(qstart[1], qend[1], theN+1)
      qdguess = np.ones([1,theN+1])*.1
      tGuess = np.linspace(0,durationInitial,theN+1)
      sinTorque = np.sin(2*np.pi*(1/durationInitial)*tGuess)

      opti.set_initial(q1, q1guess)
      opti.set_initial(q2, q2guess)
      opti.set_initial(dqdt[0,:], qdguess)
      opti.set_initial(dqdt[1,:], qdguess)
      opti.set_initial(u[0,:], sinTorque)
      opti.set_initial(u[1,:], sinTorque)

    else:#complicated guess. minjerk trajectory in q. 
      # construct minjerk guesses
      tGuess = np.linspace(0,durationInitial,theN+1)
      mj = minjerk(tGuess)

      qGuess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        qGuess[qloop,:] = qCON0[qloop] + mj*(qCON1[qloop]-qCON0[qloop])
      opti.set_initial(q, qGuess)
      
      dqdtGuess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        dqdtGuess[qloop,:] = np.gradient(qGuess[qloop,:],tGuess)
      opti.set_initial(dqdt, dqdtGuess)

      ddqdt2Guess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        ddqdt2Guess[qloop,:] = np.gradient(dqdtGuess[qloop,:],tGuess)
      opti.set_initial(ddqdt2, ddqdt2Guess)

      uGuess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        for tloop in range(0,q.shape[1]):
          uGuess[:,tloop:tloop+1] = self.inverseDynamics(qGuess[:,tloop],dqdtGuess[:,tloop],ddqdt2Guess[:,tloop])
      opti.set_initial(u, uGuess)

      dudtGuess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        dudtGuess[qloop,:] = np.gradient(uGuess[qloop,:],tGuess)
      opti.set_initial(dudt, dudtGuess)

      ddudt2Guess = np.zeros([q.shape[0],q.shape[1]])
      for qloop in range(0,q.shape[0]):
        ddudt2Guess[qloop,:] = np.gradient(dudtGuess[qloop,:],tGuess)
      opti.set_initial(ddudt2, ddudt2Guess) 
    try:
      sol = opti.solve()
    
    ############################################################################################################################################
    ############## Post optimization ############## 
    # Extract the optimal states and controls.
      optTraj = optTrajectories(solved=True)
      optTraj.time      = sol.value(time)
      optTraj.Q         = sol.value(q)
      optTraj.QDot      = sol.value(dqdt)
      optTraj.U         = sol.value(u)
      optTraj.mechPower = sol.value(mechPower)
      optTraj.costJ     = sol.value(costJ)
      optTraj.costTime  = sol.value(costTime)
      optTraj.costWork  = sol.value(costWork)
      optTraj.costFR    = sol.value(costFR)
      optTraj.uraterate = sol.value(ddudt2)
      optTraj.duration  = sol.value(duration)

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
      return optTraj, optiParamReturn(opti,Q = Q, q = q, theQStart = qstart, theQEnd = qend, dqdt = dqdt,\
       ddqdt2 = ddqdt2,u = u, dudt = dudt, ddudt2 = ddudt2, costJ = costJ, costFR = costFR, costWork = costWork, \
         costTime = costTime, timeValuation = timeValuation, N = theN, time = time)
    
    # if the optimizer fails, do something else 
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      failTraj = optTrajectories(solved = False)
      failTraj.time      = opti.debug.value(time)
      failTraj.Q         = opti.debug.value(q)
      failTraj.QDot      = opti.debug.value(dqdt)
      failTraj.U         = opti.debug.value(u)
      failTraj.mechPower = opti.debug.value(mechPower)
      failTraj.costJ     = opti.debug.value(costJ)
      failTraj.costTime  = opti.debug.value(costTime)
      failTraj.costWork  = opti.debug.value(costWork)
      failTraj.costFR    = opti.debug.value(costFR)
      failTraj.uraterate = opti.debug.value(ddudt2)
      failTraj.duration  = opti.debug.value(duration)
      #print("duration at failure:" + str(np.round(curTime)))
      return failTraj, optiParamReturn(opti)


  def guessWithTrajectory(self,op1:optiParamReturn, op2:optiParamReturn):
    
    return 0
  
  # def movementTimeOptSetup(self, 
  # theN=100, 
  # theFRCoef = 8.5e-2, 
  # theTimeValuation = 1, 
  # theDuration = [], 
  # theDurationGuess = .5):
  # Trajectory optimization problem formulation, hermite simpson, implicit dynamics constraints means
  # We ask the solver to find (for example) accelerations and write the equations of motion implicitly. 
  def movementTimeOptSetup(self, 
    theN              =100, 
    theFRCoef         = 8.5e-2, 
    theTimeValuation  = 1, 
    theDuration       = [], #if empty, we are optimizing for duration. 
    theDurationGuess  = .5):

    opti = ca.Opti()
    
    ### Define STATE (Q), Acceleration (ddqdt2), force-rate, SLACKVARS, 
    ### and Parameters: qstart, qend, timeValuation, frCoef 
    Q = opti.variable(self.DoF*4, theN+1)
    # optimal acceleration, for implicit equations of motion.
    ddqdt2 = opti.variable(self.DoF,theN+1)
    # Force rate
    ddudt2 = opti.variable(self.DoF, theN+1)
    # slack variables
    slackVars = opti.variable(self.DoF*3, theN+1)  #fully-actuated, constrain 1:fr_p, 2:fr_n, 3:power_p.
    # parameters: these can change from opt to opt
    qstart = opti.parameter(self.DoF,1)
    qend = opti.parameter(self.DoF,1)
    timeValuation = opti.parameter()
    opti.set_value(timeValuation, theTimeValuation)
    frCoef = opti.parameter()
    opti.set_value(frCoef, theFRCoef)
    ###/
    
    ### Define movement duration as either optimized, or fixed. 
    ### if we are optimizing movement time, make it an opti.variable()
    ###, and solve for it. place some loose bounds on duration.
    if not(theDuration):
      duration = opti.variable() 
      opti.subject_to(duration > 0.0)  # critical!
      opti.subject_to(duration <=20.0) # maybe unnecessary! =)
      durationInitial = theDurationGuess
      opti.set_initial(duration,durationInitial)
    else:
      duration = theDuration
      durationInitial = duration 
    dt = (duration)/theN
    time = ca.linspace(0., duration, theN+1)  # Discretized time vector
    ###/

    # extract columns of Q for handiness.
    # position
    q = Q[0:2,:]
    q1 = Q[0, :]
    q2 = Q[1, :]
    # velocity
    dqdt = Q[2:4,:]
    # force
    u = Q[4:6,:]
    # force rate
    dudt = Q[6:8,:]
    
    # Calculus equation constraint
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete nodes and enforce calculus constraints. 
    HermiteSimpsonImplicit(opti,qd,self.implicitEOMQ,Q,ddqdt2,ddudt2,dt,[2,3])
    
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

    mechPower = self.jointPower(dqdt,u)
    # positive power constraints
    opti.subject_to(time[:] >= 0.) 
    opti.subject_to(pPower[0,:] >= 0.) 
    opti.subject_to(pPower[1,:] >= 0.) 
    opti.subject_to(pPower[0,:] >= mechPower[0,:]) 
    opti.subject_to(pPower[1,:] >= mechPower[1,:]) 
    # fraterate constraints
    opti.subject_to(pddu[0,:] >= 0.)  
    opti.subject_to(pddu[1,:] >= 0.)  
    opti.subject_to(pddu[0,:] >= ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= ddudt2[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY CONSTRAINTS) ####################################
    # Boundary constraints. Often zeros
    def initAndEndZeros(opti,list):
      for var in list:
        for dof in range(0,var.shape[0]):
          opti.subject_to(var[dof,0] == 0.0)
          opti.subject_to(var[dof,-1] == 0.0)
    initAndEndZeros(opti,[dqdt,u,ddudt2,ddqdt2,pddu,nddu])
    
    opti.subject_to(q[:,0] == qstart)
    opti.subject_to(q[:,-1] == qend)

    ############################################################################################################################################
    ############## OBJECTIVE ############## 
    costTime = time[-1] * timeValuation
    costWork = trapInt(time,pPower[0,:])+trapInt(time,pPower[1,:])
    costFR = frCoef * (trapInt(time,pddu[0,:]) + trapInt(time,pddu[1,:]) - trapInt(time,nddu[0,:]) - trapInt(time,nddu[1,:]))
    costJ = costTime + costWork + costFR
    # Set cost function
    opti.minimize(costJ)

    ############################################################################################################################################
    ############## Hyperparameters and solve ############## 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(time),opti.debug.value(q1),
          opti.debug.value(time), opti.debug.value(q2),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)

    return optiParamReturn(opti,Q = Q, q = q, theQStart = qstart, theQEnd = qend, dqdt = dqdt,\
       ddqdt2 = ddqdt2,u = u, dudt = dudt, ddudt2 = ddudt2, costJ = costJ, costFR = costFR, costWork = costWork, \
         costTime = costTime, timeValuation = timeValuation, frCoef = frCoef, N = theN, time = time, duration = duration,mechPower = mechPower)

  def updateGuessAndSolve(self,oP:optiParamReturn, xstartnew:np.ndarray, xendnew:np.ndarray, \
    theDurationInitial   = 1.0, \
    theTimeValuation  = 1.0, \
    theGeneratePlots  = 1):
    
    # unpack the opti variables for ease of use. 
    theN    = oP.N                       # number of nodes
    opti    = oP.opti                    # opti optimization framework
    qCON0   = self.xy2joints(xstartnew) # q starting
    qCON1   = self.xy2joints(xendnew)   # q ending
    q       = oP.q
    dqdt    = oP.dqdt
    ddqdt2  = oP.ddqdt2
    u       = oP.u

    # update the parameters of the optimization
    opti.set_value(oP.qstart,         qCON0)      
    opti.set_value(oP.qend,           qCON1)
    opti.set_value(oP.timeValuation,  theTimeValuation)

    # now update the guess
    opti.set_initial(oP.duration, theDurationInitial)
    tGuess = np.linspace(0,       theDurationInitial,  theN+1)
    mj = minjerk(tGuess)

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

##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################
##############################################################################################################################################################################################
class optTrajectories:
  mechPower = np.array([0])
  Q = np.array([0])
  QDot = np.array([0])
  U = np.array([0])
  time = np.array([0])
  handspeed = np.array([0])
  uraterate = np.array([0])
  solved = []

  costJ = 0
  costFR = 0
  costWork = 0
  costTime = 0
  duration = 0
  peakhandspeed = 0
  
  def __init__(self,solved = True):
    self.solved = solved

  def generatePlots(self):
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.Q[0,:],
                            self.time, self.Q[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('segment angles [°]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
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


