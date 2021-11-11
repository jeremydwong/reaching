import numpy as np

# These functions were copied from their MATLAB counterparts.

def getJointPositions(l1,l2,q1,q2):
    jointPositions = [q1,q2]

    return jointPositions

def getJointVelocities(dq1,dq2,l1,l2,q1,q2):

    jointVelocities = [dq1,dq2] 

    return jointVelocities

def getRelativeJointAngles(q1,q2):
    relativeJointAngles = [q1,q2]
    
    return relativeJointAngles

def getRelativeJointAngularVelocities(dq1,dq2):

    relativeJointAngularVelocities = [dq1,dq2]
    
    return relativeJointAngularVelocities
