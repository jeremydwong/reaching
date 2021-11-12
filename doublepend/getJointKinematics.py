import numpy as np

# These functions were copied from their MATLAB counterparts.

def getJointPositions(l1,l2,q1,q2):
    jointPositions = [q1,q2]

    return jointPositions

def getHandPosition(q1,q2,l1,l2):
    handPosition = [l1*np.cos(q1)+l2*np.cos(q2), l1*np.sin(q1)+l2*np.sin(q2) ]

    return handPosition


def handVel(q,dq,l1,l2):

    Jac = np.matrix([[-l1*np.sin(q[0]),-l2*np.sin(q[1])],[l1*np.cos(q[0]),l2*np.cos(q[1])]])
    dqdt = np.matrix([[dq[0]],[dq[1]]])
    handVelocity = Jac*dqdt
    return handVelocity

def handSpeed(q,dq,l1,l2):
    temp = handVel(q,dq,l1,l2)
    return np.sqrt(temp[0]**2+temp[1]**2)

def getRelativeJointAngles(q1,q2):
    relativeJointAngles = [q1,q2]
    
    return relativeJointAngles

def getRelativeJointAngularVelocities(dq1,dq2):

    relativeJointAngularVelocities = [dq1,dq2]
    
    return relativeJointAngularVelocities
