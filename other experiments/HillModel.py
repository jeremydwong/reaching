import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import SimpleOpt as so

# decisions
# deal with lists of muscles. loop through and return dldt. 
# 

class optiHill(so.optiParam):
  lcerel    = []
  act       = []
  dynDeriv  = []

class hillModel():
  DoF = 1               # degrees of freedom
  k_FR = 1e-2           # force rate
  k_Work = 4.2          # work
  timeValuation = 1     # time valuation
  g        = 9.81          # gravity
  l_COM    = .5            # length to COM. 
  k_SE     = 1e4           # series elastic element stiffness.
  MomArm   = .03           # moment arm.

  def forceVelocity(self,F):
      return 0
  
  def forceLength(self,lrel):
    return 0
  
  def dactivation_dt(self,act,stim):
    tau = .1
    return 1/tau*(act-stim)

  def stated(self, t, state)->ca.MX:
    stated = 0
    return stated
  
  def getLengthMuscleAndTendonFromQ(self,q):
    return 0
  
  def implicitEOMQ(self, Q, acc):
    return 0
  

class kistemakerDeGroote(hillModel):
  def forceLength(self, lrel):
    Fmin = 1e-2
    cgauss2and4 = [0.406204464574880,   0.447882024677048] #values optimized for w=.56
    forcelength01 = ((1-Fmin)/2) *ca.exp(-((lrel-1)/cgauss2and4[0])**2) + ...
    ca.exp( -((lrel-1) / cgauss2and4[1] )**4) + Fmin
    return forcelength01

  # avoid the shenanegans from VU and go KU (2016 [Degroote, ..., Fregly]. Evaluation...) Instead.
  def forceVelocity(self, v):
    d = [ -0.20122668, -11.97218865,  -5.39297821,   0.51972586]
    d[0] * ca.log((d[1]* v + d[2]) + ca.sqrt((d[1]* v + d[2])**2 + 1)) + d[3]
  
  def dactivation_dt(self, act, stim):
    return super().dactivation_dt(act, stim)

  def stated(self, t, state) -> ca.MX:
    return super().stated(t, state)

  def getLengthMuscleAndTendonFromQ(self, q):
    
    return 0 

  def tendonExInMFromAngle(self,angle):
    return 0 

  def forceFromAngle(self,angle):
    self.tendonExInMFromAngle(angle) * self.k_SE * self.MomArm

  def implicitEOMQ(self, Q, dynDeriv):
    ddqdt2 = dynDeriv[0]
    dldt = dynDeriv[1]
    e1 = ddqdt2 - self.g/self.l * ca.sin(Q[0])
    #q[3] -> activation
    e2 = Q[3] * self.FMax * self.forceLength(Q[0]) * self.forceVelocity(self,dldt) - self.tendonFromAngle()
    return ca.vertcat(e1,e2)

  

  def basicHill(self, 
    theN        = 100, 
    theDuration = []) -> so.optiParam:

    #### the casadi instance of Opti helper class. 
    opti = ca.Opti()
    oP   = optiHill(opti, N = theN) # we attach all symbolic opt variables to oP, to be returned.
    
    #### STATE (Q), Acceleration (ddqdt2), force-rate-rate
    oP.Q         = opti.variable(4, theN+1)               # phi, phidot, length, activation
    
    oP.dynDeriv  = opti.variable(3,   theN+1)             # acceleration, dldt, activation.
    oP.ddudt2    = opti.variable(self.DoF,   theN+1)      # derivative of activation (fraterate)

    #### slack variables
    slackVars = opti.variable(3, theN+1)                  # for computing positive power, and +/- frr.
    
    #### parameters: 
    # these can change from opt to opt without re-setting up the optimization.
    oP.qstart        = opti.parameter(self.DoF,1)
    oP.qend          = opti.parameter(self.DoF,1)
    oP.timeValuation = opti.parameter()
    oP.kFR           = opti.parameter()
    # set initial values
    opti.set_value(oP.timeValuation, self.timeValuation)
    opti.set_value(oP.kFR, self.k_FR)
    ###/ END State, acceleration (implicit method), force-rate, slack vars.
    
    #### Define movement duration as either optimized, or fixed. 
    # Then this code can be used for both types of optimizations. 
    if not(theDuration):                                  ### FIRST: optimized param: make it an opti.variable()
      oP.duration = opti.variable() 
      opti.subject_to(oP.duration >   0.0)                # critical!
      opti.subject_to(oP.duration <= 20.0)                # maybe unnecessary! =)
      opti.set_initial(oP.duration,1.0)
      oP.time = ca.linspace(0., oP.duration, theN+1)      # Discretized time vector
      opti.subject_to(oP.time[:] >= 0.) 
    else:                                                 ### SECOND: fixed duration.
      oP.duration = theDuration
      oP.time     = ca.linspace(0., oP.duration, theN+1)  # Discretized time vector
    dt = (oP.duration)/theN

    # extract columns of Q for handiness.
    oP.q       = oP.Q[self.DoF*0 : self.DoF*1, :] # position    
    oP.dqdt    = oP.Q[self.DoF*1 : self.DoF*2, :] # velocity
    oP.lcerel  = oP.Q[self.DoF*2 : self.DoF*3, :] # length
    oP.act     = oP.Q[self.DoF*3 : self.DoF*4, :] # dact
    
    # Calculus equation constraint
    def qd(qi, dactdt, dynDeriv): 
      return ca.vertcat(qi, dynDeriv[0], dynDeriv[1], dactdt)  # dq/dt = f(q,u)
    
    #### IMPLEMENT DYNAMICS CONSTRAINTS. in subfunction, for now only HermiteSimpsonImplicit.
    so.HermiteSimpsonImplicit(opti, qd, self.implicitEOMQ, oP.Q, oP.dynDeriv, oP.ddudt2, dt, [2,3])
    
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
    initAndEndZeros(opti,[oP.dqdt, oP.u,oP.ddudt2,oP.ddqdt2,pddu,nddu])
    opti.subject_to(oP.q[:,0]   == oP.qstart)
    opti.subject_to(oP.q[:,-1]  == oP.qend)
  
    #### OBJECTIVE ##### 
    oP.costTime  = oP.time[-1] * oP.timeValuation
    oP.kWork     = self.k_FR
    oP.costWork  = oP.kWork * (so.trapInt(oP.time,oP.pPower[0,:]))
    oP.costFR    = oP.kFR *   (so.trapInt(oP.time, pddu[0,:]) +\
                               so.trapInt(oP.time,nddu[0,:]))
    oP.costJ     = oP.costTime + oP.costWork + oP.costFR
    # Set cost function
    opti.minimize(oP.costJ)

    #### Hyperparameters and plotting function #### 
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(oP.time),opti.debug.value(oP.q[0,:]),
          opti.debug.value(oP.time), opti.debug.value(oP.q[1,:]),color=(1,.8-.8*i/(maxIter),1))
    opti.callback(callbackPlots)

    return oP
