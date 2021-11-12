'''
This functions returns a CasADi function, which can be called when
solving the NLP to compute the model constraint errors.
'''

import casadi
from getSystemDynamics import getSystemDynamics

def getModelConstraintErrors(
    l1, l2, m1, m2, I1, I2):

    # CasADi variables.
    # Segment angles.
    q1_MX = casadi.MX.sym('q1_MX',1)
    q2_MX = casadi.MX.sym('q2_MX',1)      
    # Segment angular velocities.
    q1dot_MX = casadi.MX.sym('q1dot_MX',1)
    q2dot_MX = casadi.MX.sym('q2dot_MX',1) 
    # Segment angular accelerations.
    q1dotdot_MX = casadi.MX.sym('q1dotdot_MX',1)
    q2dotdot_MX = casadi.MX.sym('q2dotdot_MX',1) 
    # Joint torques.
    t1_MX = casadi.MX.sym('t1_MX',1)
    t2_MX = casadi.MX.sym('t2_MX',1)     

    # The equations of motion were described symbolically.
    constraintErrors = getSystemDynamics(
        q1_MX,q2_MX,
        q1dot_MX,q2dot_MX,
        q1dotdot_MX,q2dotdot_MX,
        t1_MX,t2_MX,
        l1,l2,
        m1,m2,
        I1,I2)
    
    # CasADi function describing implicitly the model constraint errors.
    # f(q, dq, ddq, T) == 0.
    f_getModelConstraintErrors = casadi.Function(
        'f_getModelConstraintErrors',[
        q1_MX,q2_MX,
        q1dot_MX,q2dot_MX,
        q1dotdot_MX,q2dotdot_MX,
        t1_MX,t2_MX],constraintErrors)
    
    return f_getModelConstraintErrors
