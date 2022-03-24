# critial rewriting of jer's pointmass to get rid of the jaggies. 
# approach: 1 replace with Hermite Simpson. 
# approach: 2 get rid of torque and accel both, which may have cause problems. 
#%%
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
##%%
# % Model: physical parameters.
# Mass
m1 = 1

# % Model: dynamics written implicitly.
# f_getModelConstraintErrors = getModelConstraintErrors(
#     m1,
#     l1, l2)

# % Trajectory optimization problem formulation.

moveTime = 1                            # Stride time (s)
dt = 0.01                               # Mesh size
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

# Other opt variables: frr, slack, path constraints
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


### CONSTRAINTS
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

### CONSTRAINTS
# these constraints all have to do with slack variables. 
# as long as pPower and positive/negative fr are there, 
# these should be both managed to track the fluctuating power
# and fr, and not an issue. 
opti.subject_to(pPower1 >= 0)  # bounded control
opti.subject_to(pPower2 >= 0)  # bounded control
opti.subject_to(pPower1 >= power1)  # bounded control
opti.subject_to(pPower2 >= power2)  # bounded control
opti.subject_to(pddu1 >= 0)  # bounded control
opti.subject_to(pddu2 >= 0)  # bounded control
opti.subject_to(pddu1 >= ddu1)  # bounded control
opti.subject_to(pddu2 >= ddu2)  # bounded control
opti.subject_to(nddu1 <= 0)  # bounded control
opti.subject_to(nddu2 <= 0)  # bounded control
opti.subject_to(nddu1 <= ddu1)  # bounded control
opti.subject_to(nddu2 <= ddu2)  # bounded control

# objective
J = 0.0
frCoef = 8.5e-2
def trapInt(t,inVec):
        sumval = 0.
        for ii in range(0,t.size-1):
            sumval = sumval + (inVec[ii]+inVec[ii+1])/2.*(t[ii+1]-t[ii])
        return sumval

J = trapInt(time,pPower1)+trapInt(time,pPower2)+ frCoef * (trapInt(time,pddu1) + trapInt(time,pddu2) -trapInt(time,nddu1)-trapInt(time,nddu2))
# jerk ish
#J = trapInt(time,pPower1)+trapInt(time,pPower2) + frCoef * ca.sum2(pddu1**2 + pddu2**2) + frCoef * ca.sum2(nddu1**2 + nddu2**2)


### CONSTRAINTS: 
# Boundary constraints - periodic gait.
#r = opti.variable(1, N+1)
#dr = opti.variable(1,N+1)
r = 1
dr = 0 ## a = 17.9 # for circumfererence # e = 0.9
theta =  opti.variable(1, N+1)
dtheta = opti.variable(1,N+1)

# CONSTRAINT: satisfy unit circle
opti.subject_to(r*ca.cos(theta)==q1)
opti.subject_to(r*ca.sin(theta)==q2)

# the relationship between x, y, and the radius. 
#opti.subject_to(q1**2+q2**2 == 1)

#opti.subject_to(r - a*(1-e**2)/(1-e*ca.cos(theta))==0.0)
opti.subject_to(dq1[0] == 0)
opti.subject_to(dq1[-1] == 0)

# opti.subject_to(dq2[0] == 2*ca.pi*r/moveTime)
# opti.subject_to(dq2[-1] == 2*ca.pi*r/moveTime)

opti.subject_to(q1[0]==r)
opti.subject_to(q2[0]==0.)
opti.subject_to(q1[-1]==r)
opti.subject_to(q2[-1]==0.)
# final timepoint constraint. 
#opti.subject_to(u[:, N] == 0.)


######################### 
# Set cost function
opti.minimize(J)

# Create an NLP solver and pass it configuration options, like callback plot.
maxIter = 1000
pOpt = {"expand":True}
sOpt = {"max_iter": maxIter}
opti.solver('ipopt',pOpt,sOpt)
def callbackPlots(i):
    plt.plot(opti.debug.value(q1),opti.debug.value(q2),color=(1,1-i/maxIter,1))
opti.callback(callbackPlots)

#########################
# GUESS
# When no initial guess is provided, numerical zero is assumed.
f = 1/moveTime
opti.set_initial(q1, r*ca.cos(2*ca.pi*f*time))
opti.set_initial(q2, r*ca.sin(2*ca.pi*f*time))


# Solve the NLP.
sol = opti.solve()
########################################################################
################################# post optimization collection of variables + plot
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
r_opt = sol.value(r)
theta_opt = sol.value(theta)

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