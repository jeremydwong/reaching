
# Double pendulum: get it working in python
# HERMITE SIMPSON: extended for implicit formulation.
# notes: solved 2022-03-08 the implicit failure. 
# found a bug with the dynamics-> error in concept of variable vs name of variable.

#%%
#%config InlineBackend.figure_formats = ['svg']

import casadi as ca
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pygsheets

#from generateAnimation import generateAnimation

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

# System parameters.
parms = Struct()
parms.add("m",np.array([[1.],[1.],[0.]]))
parms.add("d",np.array([[.15],[.15]])) # distance from joint axis to COM.
parms.add("l",np.array([[.3],[.3]])) # length of the segment. 
parms.add("I",np.array([[1/12],[1/12]])) # length of the segment. 
parms.add("g",0.)

# % Trajectory optimization problem formulation.
# Create opti instance.
opti = ca.Opti()

moveTime = 0.3
dt = 0.01                               # Mesh size
N = int(moveTime/dt)                  # Number of mesh intervals
time = ca.linspace(0., moveTime, N+1)  # Discretized time vector

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

acc = opti.variable(2,N+1)

qdim = q.shape[0]
ddtCon = opti.variable(qdim,N+1)
accdim = acc.shape[0]
eomCon = opti.variable(accdim,N+1)

# Other continuous opt variables:
# Segment angular accelerations.
uraterate = opti.variable(2, N+1)
# Forces/torques
# Fraterate
ddu1 = uraterate[0, :]
ddu2 = uraterate[1, :]

slackVars = opti.variable(8, N+1)
pPower1 = slackVars[0, :]
pPower2 = slackVars[1, :]

pddu1 = slackVars[2,:]
pddu2 = slackVars[3,:]

nddu1 = slackVars[4,:]
nddu2 = slackVars[5,:]

# EOM DEFINITION. 
def EOMDP(qin,accin,parms):
  # m*a - F =0. 
  g = parms.g
  l1 = parms.l[0]
  l2 = parms.l[1]
  m1 = parms.m[0]
  m2 = parms.m[1]
  m3 = parms.m[2]
  d1 = parms.d[0]
  d2 = parms.d[1]
  I1 = parms.I[0]
  I2 = parms.I[1]

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

# # EOM DEFINITION. 
# def EOM(qin,accin,parms):
#   # m*a - F =0. 
#   force0 = qin[4]
#   force1 = qin[5]
#   acc0 = accin[0]
#   acc1 = accin[1]
#   return ca.vertcat(acc0*parms.m - force0,acc1*parms.m - force1)
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
    opti.subject_to(theEOMFun(theQ[:,k],theQDotDot[:,k],parms) == 0) 

    # outDFCon[:,k] = theQ[:, k+1] - theQ[:, k] - dt/6 * (f + 4*fhalf + fnext)
    # outQDDCon[:,k] = theEOMFun(theQ[:,k],theQDotDot[:,k],parms)


HermiteSimpson(opti,qd,EOMDP,q,acc,uraterate,[2,3])

# CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
# variables will be bounded between +/- Inf).
opti.subject_to(opti.bounded(-10, q1, 10))
opti.subject_to(opti.bounded(-10, q2, 10))

### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
# 4 power constraints

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
        for ii in range(0,t.size -1):
            sumval = sumval + (inVec[ii]+inVec[ii+1])/2.*(t[ii+1]-t[ii])
        return sumval
    
J = 0.0
frCoef = 8.5e-2
J = trapInt(time,pPower1)+trapInt(time,pPower2)+ frCoef * (trapInt(time,pddu1) + trapInt(time,pddu2) - trapInt(time,nddu1) - trapInt(time,nddu2))
# jerk ish objective below:
#J = trapInt(time,pPower1)+trapInt(time,pPower2) + frCoef * ca.sum2(pddu1**2 + pddu2**2) + frCoef * ca.sum2(nddu1**2 + nddu2**2)


### CONSTRAINTS: TASK-SPECIFIC (BOUNDARY)
# Boundary constraints - periodic gait.
q1_con_start = 0.1
q1_con_end = 0.2
q2_con_start = 0
q2_con_end = 0
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

#% initial guess
q1guess = np.linspace(q1_con_start, q1_con_end, N+1)
q2guess = np.linspace(q2_con_start, q2_con_end, N+1)
opti.set_initial(q1, q1guess)
opti.set_initial(q2, q2guess)


# Set cost function
opti.minimize(J)

# Create an NLP solver.
maxIter = 1000
pOpt = {"expand":True}
sOpt = {"max_iter": maxIter}
opti.solver('ipopt',pOpt,sOpt)
def callbackPlots(i):
    plt.plot(time,opti.debug.value(q1),
      time, opti.debug.value(q2),color=(1,1-i/(maxIter),1))
opti.callback(callbackPlots)

# Solve the NLP.
sol = opti.solve()
################################# post optimization
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

# % Plots.
generate_plots = 1
if generate_plots:
    # Joint torques.
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, u1_opt,
                          time, u2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torques [Nm]')
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()

    # Relative segment angles.
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, q1_opt,
                          time, q2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Relative joint angles [°]')
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()

    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, dq1_opt,
                          time, dq2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('speed [m/s]')
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()

    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, u1_opt,
                          time, u2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()


    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, ddu1_opt,
                          time, ddu2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('force rate rate [N/s^2]')
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()

    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time, power1_opt, 
                          time, pPower1_opt, 
                          )
    plt.xlabel('Time [s]')
    plt.ylabel('Power')
    plt.legend(iter(lineObjects), ('power', 'pos'))
    plt.draw()

# %%
urrsol = opti.debug.value(uraterate)
qsol = opti.debug.value(q)
accsol = opti.debug.value(acc)

errorsGaps = np.ndarray([q.shape[0],N+1])
errorsEOM = np.ndarray([2,N+1])
def qdnp(qi, ui,accinput): 
  return np.array([qi[2], qi[3], accinput[0], accinput[1], qi[6],qi[7],ui[0],ui[1]])  # dq/dt = f(q,u)

for k in range(N):
    # Hermite Simpson qsoluadrature for all derivatives. 
    f = qdnp(qsol[:, k], urrsol[:, k], accsol[:,k])
    fnext = qdnp(qsol[:, k+1], urrsol[:, k+1], accsol[:,k])
    qhalf = 0.5*(qsol[:, k] + qsol[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(urrsol[:, k] + urrsol[:, k+1])
    acchalf = 0.5*(accsol[:, k] + accsol[:, k+1]) + dt/8*(f[2:4] - fnext[2:4])
    fhalf = qdnp(qhalf, uhalf, acchalf)
    errorsGaps[:,k] = qsol[:, k+1] - qsol[:, k] - dt/6 * (f + 4*fhalf + fnext)  # close the gaps
    ### CONSTRAINTS: IMPLICIT
    temp = np.array(EOMDP(qsol[:,k],accsol[:,k],parms))
    errorsEOM[:,k] = temp[:,0]

f,ax = plt.subplots()
plt.plot(errorsEOM[0,:])
# %%

# %%
