'''
Modified from the ISB-collocation code, here we simplify 
by going back to a point mass.
tasks
- optimize movement time
- add force-rate cost
- add slack variable 
- consider collocation (rather than transcription)
'''

import casadi as ca

import numpy as np
import matplotlib.pyplot as plt
import pygsheets

from getModelConstraintErrors import getModelConstraintErrors
from eulerIntegrator import eulerIntegrator
from getJointKinematics import getJointPositions
from getJointKinematics import getJointVelocities
from getJointKinematics import getRelativeJointAngles
from getJointKinematics import getRelativeJointAngularVelocities
#from generateAnimation import generateAnimation

# %% Plot settings.
'''
You might no want to generate the animation and figures every single time
you run the code. Feel free to adjust the variables below accordingly.
'''
generate_animation = False
generate_plots = True

# %% Model: physical parameters.
# Mass of the segments.
m1 = 1
# Length of the segments.
l1 = 0.4
l2 = 0.4

# %% Model: dynamics written implicitly.
f_getModelConstraintErrors = getModelConstraintErrors(
    m1,
    l1, l2)

# %% Trajectory optimization problem formulation.

moveTime = 0.8                        # Stride time (s)
dt = 0.01                               # Mesh size
N = int(moveTime/dt)                  # Number of mesh intervals
time = np.linspace(0., moveTime, N+1) # Discretized time vector


# Create opti instance.
opti = ca.Opti()

# Create design variables.
#
# Backward Euler scheme:
# x(t+1) = x(t) + u(t+1)dt
#
# We define the states at N+1 mesh points (starting at k=1).
# We define the controls at N mesh points (starting at k=2)
#
# k=1   k=2   k=3             k=N   k=N+1
# |-----|-----|-----|...|-----|-----|
#
# The dynamic contraints and equations of motion are NOT enforced in k=0.
#
# States.
# Segment angles.
q1 = opti.variable(1,N+1)
q2 = opti.variable(1,N+1) 
# Segment angular velocities.
dq1 = opti.variable(1,N+1)
dq2 = opti.variable(1,N+1)  
# Controls.
# Segment angular accelerations.
ddq1 = opti.variable(1,N)
ddq2 = opti.variable(1,N)   
# Joint torques.
T1 = opti.variable(1,N)
T2 = opti.variable(1,N)
# Frate
dT1 = opti.variable(1,N)
dT2 = opti.variable(1,N)
# Fraterate
ddT1 = opti.variable(1,N)
ddT2 = opti.variable(1,N)


# Set bounds on segment angles (if not otherwise specified, design
# variables will be bounded between +/- Inf).
opti.subject_to(opti.bounded(-np.pi/2, q1, np.pi/2))
opti.subject_to(opti.bounded(-np.pi/2, q2, np.pi/2))

# Set naive initial guess for the segment angles
# When no initial guess is provided, numerical zero is assumed.
q1_init = -np.pi/8
q1_final = -np.pi/6   
q2_init = np.pi/6
q2_final = -np.pi/8
q1guess = np.linspace(q1_init, q1_final, N+1)
q2guess = np.linspace(q2_init, q2_final, N+1)
opti.set_initial(q1, q1guess)
opti.set_initial(q2, q2guess)

# Initialize the cost function (J).
J = 0

# Loop over mesh points.
# k=1   k=2   k=3             k=N   k=N+1
# |-----|-----|-----|...|-----|-----|
for k in range(N):
    # States at mesh point k.
    # Segment angles.
    q1k = q1[:,k]     
    q2k = q2[:,k]     
    # Segment angular velocities.
    dq1k = dq1[:,k]   
    dq2k = dq2[:,k]   
    
    # States at mesh point k+1.
    # Segment angles.
    q1k_plus = q1[:,k+1]     
    q2k_plus = q2[:,k+1]     
    # Segment angular velocities.
    dq1k_plus = dq1[:,k+1]   
    dq2k_plus = dq2[:,k+1]   
    
    # Controls at mesh point k+1.
    # (Remember that controls are defined from k=2, so 'mesh point k+1 for
    # the states correspond to mesh point k for the controls', which is why
    # we use k and not k+1 here).
    # Segment angular accelerations.
    ddq1k_plus = ddq1[:,k] 
    ddq2k_plus = ddq2[:,k] 
    # Joint torques.
    T1k_plus = T1[:,k]     
    T2k_plus = T2[:,k]     
       
    # Stack states at mesh points k and k+1.
    Xk = ca.vertcat(q1k, q2k,   
          dq1k, dq2k)
    Xk_plus = ca.vertcat(q1k_plus, q2k_plus,
               dq1k_plus, dq2k_plus)
    
    # Stack state derivatives.
    Uk_plus = ca.vertcat(dq1k_plus, dq2k_plus, 
          ddq1k_plus, ddq2k_plus)
    

    # Path constraints - dynamic constraints.
    opti.subject_to(eulerIntegrator(Xk, Xk_plus, Uk_plus, dt) == 0)

    if k < (N-1):
        # Stack force rate
        # Stack force rate
        ddT1k_plus = ddT1[:,k]
        ddT2k_plus = ddT2[:,k]
        torquesK = ca.vertcat(T1k_plus,T2k_plus)
        torquesK_pp = ca.vertcat(T1[:,k+1],T2[:,k+1])
        dtorquesK = ca.vertcat(dT1[:,k],dT2[:,k])
        dtorquesK_pp = ca.vertcat(dT1[:,k+1],dT2[:,k+1])
        #ddtorquesK = ca.vertcat(ddT1k_plus,ddT2k_plus)
        ddtorquesK_pp = ca.vertcat(ddT1[:,k+1],ddT2[:,k+1])
        opti.subject_to(eulerIntegrator(torquesK, torquesK_pp, dtorquesK_pp, dt) == 0)
        opti.subject_to(eulerIntegrator(dtorquesK, dtorquesK_pp, ddtorquesK_pp, dt) == 0)
          
    # Path constraints - model constraints (implicit skelton dynamics).
    # We impose this error to be null (i.e., f(q, dq, ddq, T) = 0).
    modelConstraintErrors = f_getModelConstraintErrors(
        q1k_plus,q2k_plus,
        dq1k_plus,dq2k_plus,
        ddq1k_plus,ddq2k_plus,
        T1k_plus,T2k_plus)
    opti.subject_to(ca.vertcat(*modelConstraintErrors) == 0)
    
    # Cost function.
    # Minimize the weighted sum of the squared joint torques.
    #J = J + (T1k_plus**2 + T2k_plus**2)*dt
    # now we try to minimize force-rate-rate.
    J = J + (ddT1k_plus**2 + ddT2k_plus**2)*dt
    # Penalize (with low weight) segment angular accelerations for
    # regularization purposes.
        
# Boundary constraints - periodic gait.

q1_con_start = 0.1
q1_con_end = 0.2
q2_con_start = 0
q2_con_end = 0

q1_end = q1[:,-1]     
q2_end = q2[:,-1] 
q1_start = q1[:,0]      
q2_start = q2[:,0] 
dq1_end = dq1[:,-1]   
dq2_end = dq2[:,-1] 
dq1_start = dq1[:,0]    
dq2_start = dq2[:,0] 
ddq1_start = ddq1[:,0]
ddq2_start = ddq2[:,0]
ddq1_end = ddq1[:,-1]
ddq2_end = ddq2[:,-1]

opti.subject_to(q1_end == q1_con_end)
opti.subject_to(q2_end == q2_con_end)
opti.subject_to(q1_start == q1_con_start)
opti.subject_to(q2_start == q2_con_start)
opti.subject_to(dq1_end == 0)
opti.subject_to(dq2_end == 0)
opti.subject_to(dq1_start == 0)
opti.subject_to(dq2_start == 0)
opti.subject_to(ddq1_start == 0)
opti.subject_to(ddq2_start == 0)
opti.subject_to(ddq1_end == 0)
opti.subject_to(ddq2_end == 0)

# heelStrike_error = getHeelStrikeError(
#     I1,I2,I3,I4,I5,
#     d1,d2,d3,d4,d5,
#     dq1_min,dq2_min,dq3_min,dq4_min,dq5_min,
#     dq1_plus,dq2_plus,dq3_plus,dq4_plus,dq5_plus,
#     l1,l2,l4,l5,
#     m1,m2,m3,m4,m5,
#     q1_min,q2_min,q3_min,q4_min,q5_min,
#     q1_plus,q2_plus,q3_plus,q4_plus,q5_plus)
# opti.subject_to(ca.vertcat(*heelStrike_error) == 0)
        
# Boundary constraints - start at 'toe-off' and end at 'heel-strike'.
jointVelocitiesInit = getJointVelocities(
    dq1[:,0],dq2[:,0],
    l1,l2,
    q1[:,0], q2[:,0])
jointVelocitiesEnd = getJointVelocities(
    dq1[:,-1],dq2[:,-1],
    l1,l2,
    q1[:,-1],q2[:,-1])
opti.subject_to(jointVelocitiesInit[1] > 0)
opti.subject_to(jointVelocitiesEnd[1] < 0)

# Set cost function
opti.minimize(J)

# Create an NLP solver.
opti.solver('ipopt')

# Solve the NLP.
sol = opti.solve()

# %% Extract the optimal states and controls.
# Optimal segment angles.
q1_opt = sol.value(q1)
q2_opt = sol.value(q2)
# Optimal segment angular velocities.
dq1_opt = sol.value(dq1)
dq2_opt = sol.value(dq2)
# Optimal joint accelerations.
ddq1_opt = sol.value(ddq1)
ddq2_opt = sol.value(ddq2)
# Optimal joint torques.
T1_opt = sol.value(T1)
T2_opt = sol.value(T2)

# %% Generate an animation.
if generate_animation:
    jointPositions_opt = getJointPositions(
        l1,l2,l3,l4,l5,
        q1_opt,q2_opt,q3_opt,q4_opt,q5_opt)
    animation = generateAnimation(jointPositions_opt, dt, strideLength)

# %% Plots.
if generate_plots:
    # Joint torques.
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time[:-1],T1_opt,
                          time[:-1],T2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torques [Nm]')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()
    
    # Relative segment angles.
    relJointPos = getRelativeJointAngles(
        q1_opt,q2_opt)
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time,180/np.pi*relJointPos[0],
                          time,180/np.pi*relJointPos[1])
    plt.xlabel('Time [s]')
    plt.ylabel('Relative joint angles [°]')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()
    
    # Relative segment angular velocities.
    # relJointVel = getRelativeJointAngularVelocities(
    #     dq1_opt,dq2_opt)
    # fig = plt.figure()
    # ax = plt.gca()
    # lineObjects = ax.plot(time,180/np.pi*relJointVel[0],
    #                       time,180/np.pi*relJointVel[1],
    #                       time,180/np.pi*relJointVel[2],
    #                       time,180/np.pi*relJointVel[3],
    #                       time,180/np.pi*relJointVel[4])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Relative joint angular velocities [°/s]')
    # plt.legend(iter(lineObjects), ('stance ankle','stance knee','stance hip',
    #                                 'swing hip','swing knee'))
    # plt.draw()

# %% Maximum torque.
max_torque=np.max(np.abs(np.array([T1_opt, T2_opt])))


# %% Plot graphs
plt.show()
