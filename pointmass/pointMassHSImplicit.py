# Double pendulum: get it working in python
# HERMITE SIMPSON: extended for implicit formulation.
# notes: solved 2022-03-08 the implicit failure. 
# found a bug with the dynamics-> error in concept of variable vs name of variable.

#%%
%config InlineBackend.figure_formats = ['svg']

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
parms.add("m",np.array([[1],[1]]))
parms.add("d",np.array([[1],[1]])) # distance from joint axis to COM.
parms.add("l",np.array([[1],[1]])) # length of the segment. 

# % Trajectory optimization problem formulation.

moveTime = 0.3                        # Stride time (s)
dt = 0.005                               # Mesh size
N = int(moveTime/dt)                  # Number of mesh intervals
time = np.linspace(0., moveTime, N+1)  # Discretized time vector


# Create opti instance.
opti = ca.Opti()

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

# Other continuous opt variables:
# Segment angular accelerations.
fraterate = opti.variable(8, N+1)
# Forces/torques
# Fraterate
ddu1 = fraterate[0, :]
ddu2 = fraterate[1, :]

slackVars = opti.variable(8, N+1)
pPower1 = slackVars[0, :]
pPower2 = slackVars[1, :]

pddu1 = slackVars[2,:]
pddu2 = slackVars[3,:]

nddu1 = slackVars[4,:]
nddu2 = slackVars[5,:]

# EOM DEFINITION. 
def EOM(qin,accin,parms):
  # m*a - F =0. 
  force0 = qin[4]
  force1 = qin[5]
  acc0 = accin[0]
  acc1 = accin[1]
  return ca.vertcat(acc0*parms.m - force0,acc1*parms.m - force1)

# Calculus equations
def qd(qi, ui,acc): return ca.vertcat(qi[2], qi[3], acc[0], acc[1], qi[6],qi[7],ui[0],ui[1])  # dq/dt = f(q,u)
# Loop over discrete time
for k in range(N):
    # Hermite Simpson quadrature for all derivatives. 
    f = qd(q[:, k], fraterate[:, k], acc[:,k])
    fnext = qd(q[:, k+1], fraterate[:, k+1], acc[:,k])
    qhalf = 0.5*(q[:, k] + q[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(fraterate[:, k] + fraterate[:, k+1])
    acchalf = 0.5*(acc[:, k] + acc[:, k+1]) + dt/8*(f[2:4,:] - fnext[2:4,:])
    fhalf = qd(qhalf, uhalf, acchalf)
    opti.subject_to(q[:, k+1] - q[:, k] == dt/6 *
                    (f + 4*fhalf + fnext))  # close the gaps
    ### CONSTRAINTS: IMPLICIT
    opti.subject_to(EOM(q[:,k],acc[:,k],parms) == 0) 
  
# CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
# variables will be bounded between +/- Inf).
opti.subject_to(opti.bounded(-10, q1, 10))
opti.subject_to(opti.bounded(-10, q2, 10))

### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
# 4 power constraints
power1 = dq1*u1
power2 = dq2*u2

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
        for ii in range(0,t.size-1):
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
