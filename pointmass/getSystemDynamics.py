import numpy as np

# This function was copied from its MATLAB counterpart.

def getSystemDynamics(T1,T2,
                      ddq1,ddq2,
                      dq1,dq2,
                      l1,l2,
                      m1,
                      q1,q2):

    eq_systemDynamics = [T1-m1*ddq1,T2-m1*ddq2] 
    return eq_systemDynamics
