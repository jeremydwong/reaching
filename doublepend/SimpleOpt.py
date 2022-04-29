# SimpleModel structure
#%%
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ


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

  ##############################################################################################################################################################################################
  def implicitEOM(self, q, u, acc):
    return 0
  ##############################################################################################################################################################################################
  def implicitEOMQ(self, Q, acc):
    return 0
  ##############################################################################################################################################################################################
  def inverseDynamics(self, q, qdot, qdotdot, acc):
    return 0
  ##############################################################################################################################################################################################
  def kinematicJacobianInertias(self,q):
    return 0
  ##############################################################################################################################################################################################
  def kinematicJacobianRotationalInertias(self,q):
    return 0
  ##############################################################################################################################################################################################
  def xy2joints(self,xy):
    return 0
  ##############################################################################################################################################################################################
  # computes energy balance
  def energy(self, theQ, theQDot, theU, theT):
    nT = theQ.shape[1]
    
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
      for ij in np.arange(0,len(linjac)):
        vt = linjac[ij] @ theQDot[:,it]
        angvt = rotjac[ij] @ theQDot[:,it]
        e_k[0,it] = e_k[0,it] + 1/2 * self.m[ij] * vt.T @ vt
        e_k[0,it] = e_k[0,it] + 1/2 * self.I[ij] * angvt.T @ angvt

    energyOut = SimpleModelEnergy()
    energyOut.e_g = e_g.flatten()
    energyOut.e_k = e_k.flatten()
    energyOut.e_mechAll = e_mechAll
    energyOut.e_mech = e_mech.flatten()
    return energyOut

  ##############################################################################################################################################################################################
  def movementTimeOpt(self, theXYStart, theXYEnd, theN=100, theFRCoef = 8.5e-2, theTimeValuation = 1, theGeneratePlots = 1, theDuration = [], theDurationGuess = .5, LINEAR_GUESS = False, sol = []):
  # def movementTimeOpt(self, xystart, xyend, N=100, generate_plots = 1):
  # Trajectory optimization problem formulation, hermite simpson, implicit dynamics constraints means
  # We ask the solver to find (for example) accelerations and write the equations of motion implicitly. 
    opti = ca.Opti()

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
    Q = opti.variable(8, theN+1)
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
    
    # optimal acc. a decision variable allowing implicit equations of motion.
    ddqdt2 = opti.variable(2,theN+1)

    # Force rate
    ddudt2 = opti.variable(2, theN+1)

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
    opti.subject_to(pddu[0,:] >= ddudt2[0,:])  
    opti.subject_to(pddu[1,:] >= ddudt2[1,:])  
    opti.subject_to(nddu[0,:] <= 0.)  
    opti.subject_to(nddu[1,:] <= 0.)  
    opti.subject_to(nddu[0,:] <= ddudt2[0,:]) 
    opti.subject_to(nddu[1,:] <= ddudt2[1,:]) 
    
    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY)
    # Boundary constraints - periodic gait.
    opti.subject_to(dqdt[0,0] == 0.0)
    opti.subject_to(dqdt[1,0] == 0.0)
    opti.subject_to(dqdt[0,-1] == 0.0)
    opti.subject_to(dqdt[1,-1] == 0.0)
    opti.subject_to(u[0,0] == 0.0)
    opti.subject_to(u[1,0] == 0.0)
    opti.subject_to(u[0,-1] == 0.0)
    opti.subject_to(u[1,-1] == 0.0)

    qCON0 = self.xy2joints(theXYStart)
    qCON1 = self.xy2joints(theXYEnd)
    opti.subject_to(q[0,0] == qCON0[0])
    opti.subject_to(q[1,0] == qCON0[1])
    opti.subject_to(q[0,-1] == qCON1[0])
    opti.subject_to(q[1,-1] == qCON1[1])

    opti.subject_to(dudt[0,0] == 0.0)
    opti.subject_to(dudt[1,0] == 0.0)
    opti.subject_to(dudt[0,-1] == 0.0)
    opti.subject_to(dudt[1,-1] == 0.0)

    opti.subject_to(ddudt2[0,0] == 0.0)
    opti.subject_to(ddudt2[1,0] == 0.0)
    opti.subject_to(ddudt2[0,-1] == 0.0)
    opti.subject_to(ddudt2[1,-1] == 0.0)

    opti.subject_to(ddqdt2[0,0] == 0.0)
    opti.subject_to(ddqdt2[1,0] == 0.0)
    opti.subject_to(ddqdt2[0,-1] == 0.0)
    opti.subject_to(ddqdt2[1,-1] == 0.0)

    opti.subject_to(pddu[0,0] == 0.0)
    opti.subject_to(pddu[1,0] == 0.0)
    opti.subject_to(pddu[0,-1] == 0.0)
    opti.subject_to(pddu[1,-1] == 0.0)
    
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
    
    # if we don't have a warmstart
    if not(sol) or (not(sol.solved())):
      
      if LINEAR_GUESS:
        # do nothing fancy, initialize things just off 0. 
        q1guess = np.linspace(qCON0[0], qCON1[0], theN+1)
        q2guess = np.linspace(qCON0[1], qCON1[1], theN+1)
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
          ddudt2Guess[qloop,:] = np.gradient(uGuess[qloop,:],tGuess)
        opti.set_initial(ddudt2, ddudt2Guess)

    else:
      opti.set_initial(sol.value_variables())
      #lam_g0 = sol.value(sol.lam_g)          # this is setting the lagrange multipliers which casadi also refers to as duals
      #opti.set_initial(opti.lam_g, lam_g0)
    ############################################################################################################################################
    try:
      sol = opti.solve()
     
      ############################################################################################################################################
      ############## Post optimization ############## 
      # Extract the optimal states and controls.
      optTraj = optTrajectories()
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
      return sol, optTraj.duration, optTraj.costJ, optTraj.costWork, optTraj.costFR, optTraj.costTime, optTraj.peakhandspeed, optTraj  

    # if the optimizer fails, do something else 
    except:
      print("Caught: post-opti.solve() failed. Check either the first output, or the subsequent plotting code.\n")
      curTime = opti.debug.value(duration)
      #print("duration at failure:" + str(np.round(curTime)))
      return opti, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, optTrajectories()

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

  costJ = 0
  costFR = 0
  costWork = 0
  costTime = 0
  duration = 0
  peakhandspeed = 0

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
      plt.ylabel('Power')
      plt.legend(iter(lineObjects), ('1', '2'))
      plt.draw()
      plt.show()
