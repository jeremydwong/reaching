# SimpleModel structure
#%%
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

### OBJECTIVE 
def trapInt(t,inVec):
  sumval = 0.
  for ii in range(0,inVec.shape[1]-1):
      sumval = sumval + (inVec[ii]+inVec[ii+1])/2.0 * (t[ii+1]-t[ii])
  return sumval

def HermiteSimpsonImplicit(opti:ca.Opti, theDDTFun:callable, theEOMImplicitFun:callable, theQ, theQDotDot, theUIn,dt,indQDot=[2,3]):
  for k in range(theQ.shape[1]-1):
    # Hermite Simpson quadrature for all derivatives. 
    f = theDDTFun(theQ[:, k], theUIn[:, k], theQDotDot[:,k])
    fnext = theDDTFun(theQ[:, k+1], theUIn[:, k+1], theQDotDot[:,k])
    qhalf = 0.5*(theQ[:, k] + theQ[:, k+1]) + dt/8*(f - fnext)
    uhalf = 0.5*(theUIn[:, k] + theUIn[:, k+1])
    acchalf = 0.5*(theQDotDot[:, k] + theQDotDot[:, k+1]) + dt/8*(f[indQDot,:] - fnext[indQDot,:])
    fhalf = theDDTFun(qhalf, uhalf, acchalf)
    opti.subject_to(theQ[:, k+1] - theQ[:, k] == dt/6 * (f + 4*fhalf + fnext))  # close the gaps
    ### CONSTRAINT: Implicit dynamics, M*A - F = 0
    opti.subject_to(theEOMImplicitFun(theQ[:,k],theQDotDot[:,k]) == 0) 

class SimpleModelEnergy:
  e_k = np.array([0])
  e_g = np.array([0])
  e_mech = np.array([0])
  e_mechAll = np.array([0])

class SimpleModels:
  l = np.array([0])
  d = np.array([0])
  I = np.array([0])
  m = np.array([0])
  g = 0

  def implicitEOM(self, q, u, acc):
    return 0
  
  def handspeed(self, q, qdot):
    return 0

  def energy(self, q, u):
    return 0

  def kinematicJacobian(self,q):
    return 0

class optTrajectories:
  mechPower = np.array([0])
  Q = np.array([0])
  QDot = np.array([0])
  U = np.array([0])
  time = np.array([0])
  handspeed = np.array([0])
  uraterate = np.array([0])

  costJ = 0
  costFR = 0
  costWork = 0
  costTime = 0
  duration = 0
  peakhandspeed = 0

  def generatePlots(self):
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.Q[0,:],
                            self.time, self.Q[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('segment angles [°]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.QDot[0,:],
                            self.time, self.QDot[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('angular velocity [s^-1]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      # Joint torques.
      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.U[0,:],
                            self.time, self.U[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('Joint torques [Nm]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.uraterate[0,:],
                            self.time, self.uraterate[1,:])
      plt.xlabel('Time [s]')
      plt.ylabel('force rate rate [N/s^2]')
      plt.legend(iter(lineObjects), ('q[0]', 'q[1]'))
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.handspeed)
      plt.xlabel('Time [s]')
      plt.ylabel('Hand speed [m/s]')
      plt.draw()
      plt.show()

      fig = plt.figure()
      ax = plt.gca()
      lineObjects = ax.plot(self.time, self.mechPower[0,:], 
                            self.time, self.mechPower[1,:], 
                            )
      plt.xlabel('Time [s]')
      plt.ylabel('Power')
      plt.legend(iter(lineObjects), ('1', '2'))
      plt.draw()
      plt.show()