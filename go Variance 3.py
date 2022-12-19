#%%
import casadi as ca
import SimpleOpt as so
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
# sys.path.append('/Users/jeremy/git/python-control/')
import control as ctrl


# go Variance 1 ran a discretized system
# go Variance 2 will now run a continuous system, and i should be able to check that the
# continous

def trapInt(t,inVec):
  shp = inVec.shape
  if len(shp) == 1:
    dimdat = 0
  elif(shp[0]>shp[1]):
    dimdat = 0
  else:
    dimdat = 1

  sumval = 0.
  for ii in range(0,inVec.shape[dimdat]-1):
      sumval = sumval + (inVec[ii]+inVec[ii+1])/2.0 * (t[ii+1]-t[ii])
  return sumval
  
# 1D 
n     = 200 # total of 100 supports
nm    = n-1  # finish & hold at index nm
Bs    = 0   # damping
M     = 1   # Mass
kVar  = 1    # coefficient of variation from of positional variance from u^2.

opti = ca.Opti()
p = opti.variable(n,1)
v = opti.variable(n,1) 
F = opti.variable(n,1) 
Fdot = opti.variable(n,1) 
Fddot = opti.variable(n-1,1) 
# V11 = opti.variable(n-1,1) 
Ek = opti.variable(n-1,1) 
slackFddp = opti.variable(n-1,1) 
slackFddn = opti.variable(n-1,1) 
posPowJ = opti.variable(n,1) 
slackPosPowJ = opti.variable(n,1) 
T = opti.variable(1,1)
opti.subject_to(T==0.5)
dt  = T/n # dimensionless 

# State-space, continuous time. Point mass. allow for damping. 
A = np.array([[0, 1, 0, 0], [0, -Bs/M, 1/M, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
# position, velocity, F, Fdot
B = np.array([[0], [0], [0], [1]])
C = np.array([1, 0, 0, 0])
D = [0]

# 0. SYSTEM EQUATIONS. 
sseye = ctrl.ss(A,B,C,D)
# ssdeye = ctrl.c2d(sseye,dt)
#sseye = ctrl.ss(A, B, C, D)
#ssdeye = ctrl.c2d(sseye, Δt)[1,1] # exp(A Δt)

def matpow(inmat,n):
  inmatp = inmat
  for j in range(0,n-1):
    inmatp=inmatp @ inmat
  return inmatp

# 1 VARIANCE, continuous.
Sigma_j   = ca.MX.zeros(sseye.A.shape[0],sseye.A.shape[1])

Sigma_jp  = ca.MX.zeros(n,1)
for j in np.arange(0,n-1):
  #dΣⱼ = sseye.A * Σⱼ + Σⱼ' * sseye.A' + sseye.B*u[j]*u[j]*sseye.B'
  dSigma_j        = sseye.A @ Sigma_j + Sigma_j.T @ sseye.A.T + \
                    (F[j] * F[j] * kVar) * (sseye.B @ sseye.B.T) 
  Sigma_j         = Sigma_j + (dSigma_j)*dt           #Euler
  #Sigma_js[:,:,j] = Sigma_j
  Sigma_jp[j+1]   = Sigma_j[0,0]           #sigma position over time. 

# 2. HOLD period
for j in np.arange(nm,n): # julia nm+1:n # during hold period, keep velocity zero
    opti.subject_to(v[j] == 0)

# 3. SLACK vars for fdotdot
for j in np.arange(0,n-1): # during hold period, keep velocity zero
  opti.subject_to(slackFddp[j] >=Fddot[j])
  opti.subject_to(slackFddp[j] >=0)
  opti.subject_to(slackFddn[j] <=Fddot[j])
  opti.subject_to(slackFddn[j] <=0)

# 4. DYNAMICS, as state-space linear system
for j in np.arange(1,n): #julia 2:n
    opti.subject_to(p[j] == p[j-1] + dt*v[j-1])
    opti.subject_to(v[j] == v[j-1] + dt*(-Bs*v[j-1] + F[j-1])) #EOM
    opti.subject_to(F[j] == F[j-1] + dt*Fdot[j-1])
    opti.subject_to(Fdot[j] == Fdot[j-1] + dt*Fddot[j-1])

#5. OBJECTIVE: ENERGY need to be set here. 
for j in np.arange(0,n):#julia j in 1:n-1
  opti.subject_to(posPowJ[j]  == v[j] * F[j])
  # opti.subject_to(posPowJ[j]  == 0.5*(v[j]+v[] * F[j]) #could make power be an average like art. 
  opti.subject_to(slackPosPowJ[j] >=posPowJ[j])
  opti.subject_to(slackPosPowJ[j] >=0)

# 6. Initial conditions and final conditions
# Constants
p_0 = 0    # Initial position
v_0 = 0    # Initial velocity
p_f = 1    # Final position
v_f = 0    # Final velocity (at step nm)

opti.subject_to(p[0] == 0)
opti.subject_to(v[0] == 0)
opti.subject_to(p[nm] == p_f)
opti.subject_to(v[nm] == v_f)  # final condition is at nm
opti.subject_to(F[0] == 0)
opti.subject_to(Fdot[0] == 0)
opti.subject_to(F[nm] == 0)
opti.subject_to(Fdot[nm] == 0)


# 7 OBJECTIVES: E(W+FR) + T
tauvec    = ca.linspace(0,1,n)                              #normalized time.

# work
kPosWorkW   = 4.2
objPosWorkJ = kPosWorkW * trapInt(tauvec[:,0]*T, slackPosPowJ)

# fdd
kFR     = 1e-1                                                  #bells for 1e-1, with T = 1, work = 8
objFrJ  = kFR * ( trapInt(tauvec[0:-1,0]*T, -slackFddn) +\
                  trapInt(tauvec[0:-1,0]*T,  slackFddp))

# time
kT       = opti.parameter(1,1)
objTimeJ = kT*T

# F2
objF2 = trapInt(tauvec*T,F * F)

# Variance settings
objVarianceHold = ca.sum1(Sigma_jp[nm:])

# Variance parameters, to allow rapid reset-then-re-running of code (below).
Sigma_limit = opti.parameter(1, 1)
# V11_limit = opti.parameter(1, 1)
# opti.subject_to(Sigma_jp[-1] <= Sigma_limit)                              #sigma constraint
opti.subject_to(Sigma_jp[-1] <= Sigma_limit)                              #sigma constraint
# opti.subject_to(V11[-1] <= V11_limit)                              #sigma constraint
#minimize Fdot squared over movement period

opti.minimize(objFrJ + objPosWorkJ + objTimeJ)
# opti.minimize(objVarianceHold)


# parameters and initial guess.
opti.set_initial(T,0.5)
opti.set_value(kT,30)
# disengage the variation limits. 
opti.set_value(Sigma_limit,1e-3)
# opti.set_value(V11_limit,1)

maxIter = 1000
pOpt = {"expand":True}
sOpt = {"max_iter"        : maxIter,
        "constr_viol_tol" : 1e-2,
        "dual_inf_tol"    : 1e-2}
opti.solver('ipopt',pOpt,sOpt)
f,ax = plt.subplots(2,2)
def callbackPlots(i):
    tt = opti.debug.value(T)
    ax[0,0].plot(tauvec*tt,opti.debug.value(p))
    ax[0,1].plot(tauvec*tt,opti.debug.value(v),color=(1,.8-.8*i/(maxIter),1))
    #ax[1,0].plot(tvec[0:-1],opti.debug.value(V11),color=(1,.8-.8*i/(maxIter),1))
    ax[1,0].plot(tauvec,opti.debug.value(Sigma_jp),color=(1,.8-.8*i/(maxIter),1))
    #ax[1,0].plot(tauvec[0:-1]*tt,opti.debug.value(V11),color=(1,.8-.8*i/(maxIter),1))
    ax[1,1].plot(tauvec[0:-1]*tt,opti.debug.value(Fddot),color=(1,.8-.8*i/(maxIter),1))
opti.callback(callbackPlots)

try:
      sol = opti.solve()
except:
  print("did not find solution")

Topt = sol.value(T)
fig,ax1 = plt.subplots(2,2)
ax1[0,0].plot(tauvec[0:-1]*Topt,sol.value(slackFddn),\
  tauvec[0:-1]*Topt,sol.value(slackFddp))
ax1[0,1].plot(tauvec[0:-1]*Topt,sol.value(Fddot))
sol.value(objFrJ)
ax1[1,0].plot(sol.value(v[:]))
