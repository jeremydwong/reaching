import numpy as np
from casadi.casadi import cos
from casadi.casadi import sin
def getSystemDynamics(q1,q2,                    
                    q1dot,q2dot,
                    ddq1,ddq2,
                    T1,T2,
                    l1,l2,
                    m1,m2,
                    I1,I2):

    g = 0.0
    mHand = 0.0
    d1 = 0.5*l1
    d2 = 0.5*l2
    Fx = 0.0
    Fy = 0.0
    
    m11 = I1 + d1**2*m1 + l1**2*m2 + l1**2*mHand
    m12 = l1*cos(q1 - q2)*(d2*m2 + l2*mHand)
    m21 = m12
    m22 =m2*d2**2 + mHand*l2**2 + I2
        
    F1 = -l1*q2dot**2*sin(q1 - q2)*(d2*m2 + l2*mHand)
    F2 = l1*q1dot**2*sin(q1 - q2)*(d2*m2 + l2*mHand)
        
        # m * a - f Newton implicit
    e1 = m11*ddq1 + m12*ddq2 - F1 - T1 + T2
    e2 = m21*ddq1 + m22*ddq2 - F2 - T2

    #e1 = ddq1*m1 - T1 #debugging, override dynamics.
    #e2 = ddq2*m1 - T2 #debugging
    eq_systemDynamics = [e1,e2] 
    return eq_systemDynamics
