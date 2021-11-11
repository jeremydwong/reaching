'''
This functions returns a CasADi function, which can be called when
solving the NLP to compute the model constraint errors.
'''

import casadi
from getSystemDynamics import getSystemDynamics

def getModelConstraintErrors(
    m1, l1, l2):

    # CasADi variables.
    # Segment angles.
    q1_MX = casadi.MX.sym('q1_MX',1)
    q2_MX = casadi.MX.sym('q2_MX',1)      
    # Segment angular velocities.
    dq1_MX = casadi.MX.sym('dq1_MX',1)
    dq2_MX = casadi.MX.sym('dq2_MX',1) 
    # Segment angular accelerations.
    ddq1_MX = casadi.MX.sym('ddq1_MX',1)
    ddq2_MX = casadi.MX.sym('ddq2_MX',1)  
    # Joint torques.
    T1_MX = casadi.MX.sym('T1_MX',1)
    T2_MX = casadi.MX.sym('T2_MX',1)     
    
    # The equations of motion were described symbolically.
    constraintErrors = getSystemDynamics(
        T1_MX,T2_MX,        
        ddq1_MX,ddq2_MX,
        dq1_MX,dq2_MX,
        l1,l2,
        m1,
        q1_MX,q2_MX)
    
    # CasADi function describing implicitly the model constraint errors.
    # f(q, dq, ddq, T) == 0.
    f_getModelConstraintErrors = casadi.Function(
        'f_getModelConstraintErrors',[
        q1_MX,q2_MX,
        dq1_MX,dq2_MX,
        ddq1_MX,ddq2_MX,
        T1_MX,T2_MX],constraintErrors)
    
    return f_getModelConstraintErrors
