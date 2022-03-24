#Double Pendlum Class
from calendar import c
import struct
import numpy as np
import casadi as ca
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pygsheets

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
  l = np.array([.3,.3])
  d = np.array([0.15,0.15])
  I = np.array([1/12.0, 1/12.0])
  m = np.array([2.1, 1.65, 0.0])
  g= 0.0


  def handspeed(self, q, qdot):
    l1 = self.l[0]
    l2 = self.l[1]
    Kinjac = np.array([[ -l1*np.sin(q[0]), -l2*np.sin(q[1])],\
        [l1*np.cos(q[0]),  l2*np.cos(q[1])]])
    hvel = Kinjac @ qdot
    htan = np.sqrt(hvel[0]**2+hvel[1]**2)
    return htan, hvel
      

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

  def movementTimeOpt(self, xystart, xyend, N=100, theFRCoef = 8.5e-2, theTimeValuation = 1, theGeneratePlots = 1):
  # def movementTimeOpt(self, xystart, xyend, N=100, generate_plots = 1):
  # Trajectory optimization problem formulation, hermite simpson
    
    # Create opti instance.
    opti = ca.Opti()

    # Opt variables for state. 
    duration = opti.variable() #opti.variable()
    dt = duration/N
    time = ca.linspace(0., duration, N+1)  # Discretized time vector
    q = opti.variable(8, N+1)
    # position
    q1 = q[0, :]
    q2 = q[1, :]
    # velocity
    dq1 = q[2, :]
    dq2 = q[3, :]
    # force
    u1 = q[4, :]
    u2 = q[5, :]
    # force rate
    du1 = q[6, :]
    du2 = q[7, :]

    # optimal acc. a decision variable allowing implicit equations of motion.
    acc = opti.variable(2,N+1)

    qdim = q.shape[0]
    ddtCon = opti.variable(qdim,N+1)
    accdim = acc.shape[0]
    eomCon = opti.variable(accdim,N+1)

    # Force rate
    uraterate = opti.variable(2, N+1)
    ddu1 = uraterate[0, :]
    ddu2 = uraterate[1, :]

    # slack variables for power and force-rate-rate
    slackVars = opti.variable(8, N+1)
    pPower1 = slackVars[0, :]
    pPower2 = slackVars[1, :]
    pddu1 = slackVars[2,:]
    pddu2 = slackVars[3,:]
    nddu1 = slackVars[4,:]
    nddu2 = slackVars[5,:]

    def EOM(qin,accin,parms):
      uin1 = qin[4]
      uin2 = qin[5]
      acc1 = accin[0]
      acc2 = accin[1]
      # m*a - F =0. 
      return ca.vertcat(acc1*parms.m[0] - uin1,acc2*parms.m[0] - uin2)

    power1 = dq1*u1
    power2 = dq2*u2 - u2*dq1

    # Calculus equations
    def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
    # Loop over discrete time

    def HermiteSimpson(theOpt:ca.Opti, theDDTFun, theEOMFun, theQ, theQDotDot, theUIn,indQDot=[2,3]):
      outDFCon = ca.SX(theQ.shape[0],theQ.shape[1])
      outQDDCon = ca.SX(theQ.shape[0],theQ.shape[1])
      for k in range(theQ.shape[1]-1):
        # Hermite Simpson quadrature for all derivatives. 
        f = theDDTFun(theQ[:, k], theUIn[:, k], theQDotDot[:,k])
        fnext = theDDTFun(theQ[:, k+1], theUIn[:, k+1], theQDotDot[:,k])
        qhalf = 0.5*(theQ[:, k] + theQ[:, k+1]) + dt/8*(f - fnext)
        uhalf = 0.5*(theUIn[:, k] + theUIn[:, k+1])
        acchalf = 0.5*(theQDotDot[:, k] + theQDotDot[:, k+1]) + dt/8*(f[indQDot,:] - fnext[indQDot,:])
        fhalf = qd(qhalf, uhalf, acchalf)
        opti.subject_to(theQ[:, k+1] - theQ[:, k] == dt/6 * (f + 4*fhalf + fnext))  # close the gaps
        ### CONSTRAINTS: IMPLICIT
        opti.subject_to(theEOMFun(theQ[:,k],theQDotDot[:,k]) == 0) 
        # outDFCon[:,k] = theQ[:, k+1] - theQ[:, k] - dt/6 * (f + 4*fhalf + fnext)
        # outQDDCon[:,k] = theEOMFun(theQ[:,k],theQDotDot[:,k],parms)

    HermiteSimpson(opti,qd,self.implicitEOM,q,acc,uraterate,[2,3])
    
    # CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
    # variables will be bounded between +/- Inf).
    opti.subject_to(opti.bounded(-10, q1, 10))
    opti.subject_to(opti.bounded(-10, q2, 10))

    ### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
    # positive power constraints
    opti.subject_to(pPower1 >= 0.) 
    opti.subject_to(pPower2 >= 0.) 
    opti.subject_to(pPower1 >= power1) 
    opti.subject_to(pPower2 >= power2) 
    # fraterate constraints
    opti.subject_to(pddu1 >= 0.)  
    opti.subject_to(pddu2 >= 0.)  
    opti.subject_to(pddu1 >= ddu1)  
    opti.subject_to(pddu2 >= ddu2)  
    opti.subject_to(nddu1 <= 0.)  
    opti.subject_to(nddu2 <= 0.)  
    opti.subject_to(nddu1 <= ddu1) 
    opti.subject_to(nddu2 <= ddu2) 
    #opti.subject_to(slackVars[:, N] == 0.)

    ### OBJECTIVE
    def trapInt(t,inVec):
            sumval = 0.
            for ii in range(0,N):
                sumval = sumval + (inVec[ii]+inVec[ii+1])/2.0*(dt)
            return sumval
        
    frCoef = theFRCoef
    timeValuation = theTimeValuation
    costTime = duration * timeValuation
    costWork = trapInt(time,pPower1)+trapInt(time,pPower2)
    costFR = frCoef * (trapInt(time,pddu1) + trapInt(time,pddu2) - trapInt(time,nddu1) - trapInt(time,nddu2))
    J = costTime + costWork + costFR

    #################################### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY)
    # Boundary constraints - periodic gait.
    q1CON0 = doublePendulum.xy2joints(xystart,self.l)
    q1CON1 = doublePendulum.xy2joints(xyend,self.l)
    q1_con_start = q1CON0[0]
    q2_con_start = q1CON0[1]
    
    q1_con_end = q1CON1[0]
    q2_con_end = q1CON1[1]

    opti.subject_to(q1[0] == q1_con_start)
    opti.subject_to(q2[0] == q2_con_start)
    opti.subject_to(q1[-1] == q1_con_end)
    opti.subject_to(q2[-1] == q2_con_end)
    opti.subject_to(dq1[0] == 0.0)
    opti.subject_to(dq2[0] == 0.0)
    opti.subject_to(dq1[-1] == 0.0)
    opti.subject_to(dq2[-1] == 0.0)
    opti.subject_to(u1[0] == 0.0)
    opti.subject_to(u2[0] == 0.0)
    opti.subject_to(u1[-1] == 0.0)
    opti.subject_to(u2[-1] == 0.0)

    ### i don't understand why these are infeasible
    opti.subject_to(du1[0] == 0.0)
    opti.subject_to(du2[0] == 0.0)
    opti.subject_to(du1[-1] == 0.0)
    opti.subject_to(du2[-1] == 0.0)

    opti.subject_to(ddu1[0] == 0.0)
    opti.subject_to(ddu2[0] == 0.0)
    opti.subject_to(ddu1[-1] == 0.0)
    opti.subject_to(ddu2[-1] == 0.0)

    opti.subject_to(acc[0,0] == 0.0)
    opti.subject_to(acc[1,0] == 0.0)
    opti.subject_to(acc[0,-1] == 0.0)
    opti.subject_to(acc[1,-1] == 0.0)

    opti.subject_to(pddu1[0] == 0.0)
    opti.subject_to(pddu2[0] == 0.0)
    opti.subject_to(pddu1[-1] == 0.0)
    opti.subject_to(pddu2[-1] == 0.0)

    opti.subject_to(duration >=0.0) #critical!
    opti.subject_to(duration <=10.0) #critical!
    
    q1guess = np.linspace(q1_con_start, q1_con_end, N+1)
    q2guess = np.linspace(q2_con_start, q2_con_end, N+1)
    opti.set_initial(q1, q1guess)
    opti.set_initial(q2, q2guess)

    opti.set_initial(duration,1.0)
    ####################################/ END CONSTRAINTS: TASK-SPECIFIC (BOUNDARY)

    # Set cost function
    opti.minimize(J)

    # Hyperparameters
    maxIter = 1000
    pOpt = {"expand":True}
    sOpt = {"max_iter": maxIter}
    opti.solver('ipopt',pOpt,sOpt)
    def callbackPlots(i):
        plt.plot(opti.debug.value(time),opti.debug.value(q1),
          opti.debug.value(time), opti.debug.value(q2),color=(1,1-i/(maxIter),1))
    opti.callback(callbackPlots)

    # Solve the NLP.
    sol = opti.solve()

    ################################# ##################################################################
    ################################# ##################################################################
    # post optimization
    
    # % Extract the optimal states and controls.
    # Optimal segment angles.
    q1_opt = sol.value(q1)
    q2_opt = sol.value(q2)
    # Optimal segment angular velocities.
    dq1_opt = sol.value(dq1)
    dq2_opt = sol.value(dq2)
    # Optimal torques.
    u1_opt = sol.value(u1)
    u2_opt = sol.value(u2)
    ddu1_opt = sol.value(ddu1)
    ddu2_opt = sol.value(ddu2)
    pPower1_opt = sol.value(pPower1)
    pPower2_opt = sol.value(pPower2)
    power1_opt = sol.value(power1)
    power2_opt = sol.value(power2)
    time_opt = sol.value(time)
    J_opt = sol.value(J)
    duration_opt = sol.value(duration)
    costWork_opt = sol.value(costWork)
    costTime_opt = sol.value(costTime)
    costFR_opt = sol.value(costFR)

    # return trajectories to 'output'
    output = Struct()
    output.add("time",time_opt)
    output.add("q1",q1_opt)
    output.add("q2",q2_opt)
    output.add("dq1",dq1_opt)
    output.add("dq2",dq2_opt)
    output.add("u1",u1_opt)
    output.add("u2",u2_opt)
    output.add("power1",power1_opt)
    output.add("power2",power2_opt)
    output.add("costTime",costTime_opt)
    output.add("costWork",costWork_opt)
    output.add("costFR",costFR_opt)
    output.add("costJ",costFR_opt)
    output.add("duration",duration_opt)

    # compute peak handspeed
    handvel_opt = np.ndarray([q1_opt.shape[0]])
    for i in range(0,time_opt.shape[0]):
      qtemp = np.array([output.q1[i],output.q2[i]])
      qdottemp = np.array([output.dq1[i],output.dq2[i]])
      handvel_opt[i],dum = self.handspeed(qtemp,qdottemp)
    output.add("handvel",handvel_opt)
    peakhandspeed = max(handvel_opt)
    output.add("peakhandspeed",peakhandspeed)
    if theGeneratePlots:
      # segment angles.
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, q1_opt,
                            time_opt, q2_opt)
      plt.xlabel('Time [s]')
      plt.ylabel('segment angles [°]')
      plt.legend(iter(lineObjects), ('x', 'y'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, dq1_opt,
                            time_opt, dq2_opt)
      plt.xlabel('Time [s]')
      plt.ylabel('speed [s^-1]')
      plt.legend(iter(lineObjects), ('x', 'y'))
      plt.draw()
      plt.show()

      # Joint torques.
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, u1_opt,
                            time_opt, u2_opt)
      plt.xlabel('Time [s]')
      plt.ylabel('Joint torques [Nm]')
      plt.legend(iter(lineObjects), ('x', 'y'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, u1_opt,
                            time_opt, u2_opt)
      plt.xlabel('Time [s]')
      plt.ylabel('Force [N]')
      plt.legend(iter(lineObjects), ('x', 'y'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, ddu1_opt,
                            time_opt, ddu2_opt)
      plt.xlabel('Time [s]')
      plt.ylabel('force rate rate [N/s^2]')
      plt.legend(iter(lineObjects), ('x', 'y'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(time_opt, power1_opt, 
                            time_opt, pPower1_opt, 
                            )
      plt.xlabel('Time [s]')
      plt.ylabel('Power')
      plt.legend(iter(lineObjects), ('power', 'pos'))
      plt.draw()
      plt.show()

    return duration_opt, J_opt, costWork_opt, costFR_opt, costTime_opt, peakhandspeed , output  