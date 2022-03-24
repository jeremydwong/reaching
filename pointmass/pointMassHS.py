# critial rewriting of jer's pointmass to get rid of the jaggies, 
# which worked. 
# This code tests hermite simpson, which also worked. 
#%%
%config InlineBackend.figure_formats = ['svg']
import casadi as ca
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pygsheets

from getModelConstraintErrors import getModelConstraintErrors
from eulerIntegrator import eulerIntegrator
from getJointKinematics import getJointPositions
from getJointKinematics import getJointVelocities
from getJointKinematics import getRelativeJointAngles
from getJointKinematics import getRelativeJointAngularVelocities
#from generateAnimation import generateAnimation

generate_animation = False
generate_plots = True

# % Model: physical parameters.
# Mass
m1 = 1

# % Model: dynamics written implicitly.
# f_getModelConstraintErrors = getModelConstraintErrors(
#     m1,
#     l1, l2)

# % Trajectory optimization problem formulation.

moveTime = 0.3                        # Stride time (s)
dt = 0.005                               # Mesh size
N = int(moveTime/dt)                  # Number of mesh intervals
time = np.linspace(0., moveTime, N+1)  # Discretized time vector


# Create opti instance.
opti = ca.Opti()

m = 1
# The dynamic contraints and equations of motion are NOT enforced in k=0.
#
# States.

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
# Other continuous opt variables:
# Segment angular accelerations.
optvars = opti.variable(8, N+1)
# Forces/torques
# Fraterate
ddu1 = optvars[0, :]
ddu2 = optvars[1, :]

pPower1 = optvars[2, :]
pPower2 = optvars[3, :]

pddu1 = optvars[4,:]
pddu2 = optvars[5,:]

nddu1 = optvars[6,:]
nddu2 = optvars[7,:]


# state derivative equations
def qd(qi, ui): return ca.vertcat(qi[2], qi[3], qi[4]/m, qi[5]/m, qi[6],q[7],ui[0],ui[1])  # dq/dt = f(q,u)
# Loop over discrete time
for k in range(N):
    # Hermite Simpson quadrature for all derivatives. 
    f = qd(q[:, k], optvars[:, k])
    fnext = qd(q[:, k+1], optvars[:, k+1])
    qhalf = 0.5*(q[:, k] + q[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(optvars[:, k] + optvars[:, k+1])
    fhalf = qd(qhalf, uhalf)
    opti.subject_to(q[:, k+1] - q[:, k] == dt/6 *
                    (f + 4*fhalf + fnext))  # close the gaps

power1 = dq1*u1
power2 = dq2*u2

# CONSTRAINTS (NON_TASK_SPECIFIC): BROAD BOX LIMITS
# variables will be bounded between +/- Inf).
opti.subject_to(opti.bounded(-10, q1, 10))
opti.subject_to(opti.bounded(-10, q2, 10))

### CONSTRAINTS (NON_TASK_SPECIFIC): SLACK VARIABLES FOR POWER
# 4 power constraints
opti.subject_to(pPower1 >= 0) 
opti.subject_to(pPower2 >= 0) 
opti.subject_to(pPower1 >= power1) 
opti.subject_to(pPower2 >= power2) 

# some number of frate constraints
opti.subject_to(pddu1 >= 0)  
opti.subject_to(pddu2 >= 0)  
opti.subject_to(pddu1 >= ddu1)  
opti.subject_to(pddu2 >= ddu2)  
opti.subject_to(nddu1 <= 0)  
opti.subject_to(nddu2 <= 0)  
opti.subject_to(nddu1 <= ddu1) 
opti.subject_to(nddu2 <= ddu2) 

# constraint for final fdotdot. seems not mandatory but changes solution. 
opti.subject_to(optvars[:, N] == 0.)

### OBJECTIVE
def trapInt(t,inVec):
        sumval = 0.
        for ii in range(0,t.size-1):
            sumval = sumval + (inVec[ii]+inVec[ii+1])/2.*(t[ii+1]-t[ii])
        return sumval
    
J = 0.0
frCoef = 8.5e-2
J = trapInt(time,pPower1)+trapInt(time,pPower2)+ frCoef * (trapInt(time,pddu1) + trapInt(time,pddu2) -trapInt(time,nddu1)-trapInt(time,nddu2))
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

opti.subject_to(pddu1[0] == 0.0)
opti.subject_to(pddu2[0] == 0.0)
opti.subject_to(pddu1[-1] == 0.0)
opti.subject_to(pddu2[-1] == 0.0)

# Set cost function
opti.minimize(J)

# Create an NLP solver.
opti.solver('ipopt')

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


# % Generate an animation.
if generate_animation:
    jointPositions_opt = getJointPositions(
        l1, l2, l3, l4, l5,
        q1_opt, q2_opt, q3_opt, q4_opt, q5_opt)
    animation = generateAnimation(jointPositions_opt, dt, strideLength)

# % Plots.
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
    plt.legend(iter(lineObjects), ('x', 'y'))
    plt.draw()

# %%
