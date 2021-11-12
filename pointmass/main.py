'''
Modified from the ISB-collocation code, here we simplify 
by going back to a point mass.
tasks
- optimize movement time
- add force-rate cost
- add slack variable 
- consider collocation (rather than transcription)

this code is based upon Wouw and Falisse's ASB tutorial on collocation. 
'''
#%%
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

# % Plot settings.
'''
You might no want to generate the animation and figures every single time
you run the code. Feel free to adjust the variables below accordingly.
'''
generate_animation = False
generate_plots = True

# % Model: physical parameters.
# Mass of the segments.
m1 = 1
# Length of the segments.
l1 = 0.4
l2 = 0.4

# % Model: dynamics written implicitly.
f_getModelConstraintErrors = getModelConstraintErrors(
    m1,
    l1, l2)

# % Trajectory optimization problem formulation.

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
ddq1 = opti.variable(1,N+1)
ddq2 = opti.variable(1,N+1)   
# Joint torques.
T1 = opti.variable(1,N+1)
T2 = opti.variable(1,N+1)
# Frate
dT1 = opti.variable(1,N+1)
dT2 = opti.variable(1,N+1)
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
    ddq1k_plus = ddq1[:,k+1] 
    ddq2k_plus = ddq2[:,k+1] 
    # Joint torques.
    T1k = T1[:,k]
    T2k = T2[:,k]
    T1k_plus = T1[:,k+1]     
    T2k_plus = T2[:,k+1]     
       
    # Stack states at mesh points k and k+1.
    Xk = ca.vertcat(q1k, q2k,   
          dq1k, dq2k)
    Xk_plus = ca.vertcat(q1k_plus, q2k_plus,
               dq1k_plus, dq2k_plus)
    
    # Stack state derivatives.
    Uk_plus = ca.vertcat(dq1k_plus, dq2k_plus, 
          ddq1[:,k], ddq2[:,k])
    

    # Path constraints - dynamic constraints.
    opti.subject_to(eulerIntegrator(Xk, Xk_plus, Uk_plus, dt) == 0)

    torquesK = ca.vertcat(T1k,T2k)
    torquesK_p = ca.vertcat(T1[:,k+1],T2[:,k+1])
    dtorquesK_p = ca.vertcat(dT1[:,k+1],dT2[:,k+1])
    opti.subject_to(eulerIntegrator(torquesK, torquesK_p, dtorquesK_p, dt) == 0)

    dtorquesK = ca.vertcat(dT1[:,k],dT2[:,k])  
    ddtorquesK_p = ca.vertcat(ddT1[:,k],ddT2[:,k])
    opti.subject_to(eulerIntegrator(dtorquesK, dtorquesK_p, ddtorquesK_p, dt) == 0)
    
    # Path constraints - model constraints (implicit skelton dynamics).
    # We impose this error to be null (i.e., f(q, dq, ddq, T) = 0).
    modelConstraintErrors = f_getModelConstraintErrors(
        q1k_plus,q2k_plus,
        dq1k_plus,dq2k_plus,
        ddq1[:,k],ddq2[:,k],
        T1[:,k],T2[:,k])
    opti.subject_to(ca.vertcat(*modelConstraintErrors) == 0)
    
    ddT1k = ddT1[:,k]
    ddT2k = ddT2[:,k]
    # Cost function.

J = 0.0
for k in range(1,N):
    J = J + ((ddT1[k]+ddT1[k-1])/2)**2 *dt + ((ddT2[k]+ddT2[k-1])/2)**2 *dt
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
opti.subject_to(dq1_end == 0.0)
opti.subject_to(dq2_end == 0.0)
opti.subject_to(dq1_start == 0.0)
opti.subject_to(dq2_start == 0.0)
opti.subject_to(ddq1_start == 0.0)
opti.subject_to(ddq2_start == 0.0)
opti.subject_to(ddq1_end == 0.0)
opti.subject_to(ddq2_end == 0.0)

      
# Set cost function
opti.minimize(J)

# Create an NLP solver.
opti.solver('ipopt')

# Solve the NLP.
sol = opti.solve()

# % Extract the optimal states and controls.
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

ddT1_opt = sol.value(ddT1)
ddT2_opt = sol.value(ddT2)

# % Generate an animation.
if generate_animation:
    jointPositions_opt = getJointPositions(
        l1,l2,l3,l4,l5,
        q1_opt,q2_opt,q3_opt,q4_opt,q5_opt)
    animation = generateAnimation(jointPositions_opt, dt, strideLength)

# % Plots.
if generate_plots:
    # Joint torques.
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time,T1_opt,
                          time,T2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torques [Nm]')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()
    
    # Relative segment angles.
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time,q1_opt,
                          time,q2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('Relative joint angles [°]')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()
    
    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time,dq1_opt,
                          time,dq2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('speed [m/s]')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()

    fig = plt.figure()
    ax = plt.gca()
    lineObjects = ax.plot(time[:-1],ddT1_opt,
                          time[:-1],ddT2_opt)
    plt.xlabel('Time [s]')
    plt.ylabel('force rate')
    plt.legend(iter(lineObjects), ('x','y'))
    plt.draw()

# %% Maximum torque.
max_torque=np.max(np.abs(np.array([T1_opt, T2_opt])))


# %% Plot graphs
plt.show()
