# %% [markdown]
# ## Force-rate for double-integrator eye with hold
# The hold period makes it trickier to optimize the movement. It's quite a bit easier not to have the hold. We don't really get bell shape, need to modify final conditions.

# %%
import casadi as ca
import SimpleOpt as so
import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
# sys.path.append('/Users/jeremy/git/python-control/')
import control as ctrl

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

# Constants

CONCYCLIC1 = 1
n = 300 # total of 100 ms
ih = n-1 # finish at ind ih

kVar = 1
Bs = 0
M = 1
eyefrresults = []

# Create JuMP model, using Ipopt as the solver
opti = ca.Opti()

p = opti.variable(n,1)
v = opti.variable(n,1) 
F = opti.variable(n,1) 
Fdot = opti.variable(n,1) 
Fddot = opti.variable(n-1,1) 
slackFddp = opti.variable(n-1,1) 
slackFddn = opti.variable(n-1,1) 
posPowJ = opti.variable(n,1) 
slackPosPowJ = opti.variable(n,1) 
u = opti.variable(n,1)
T = opti.variable(1,1)
opti.subject_to(T==0.5)

dt = T/n # in s

tc = 0.1
#A = [0 1 0 0; 0 -damp/M 1/M 0; 0 0 0 1; 0 0 0 -1/tc] # position, velocity, F, Fdot just with damping
A = np.array([[0, 1, 0, 0], [0, -Bs/M, 1/M, 0], [0, 0, -1/tc, 0], [0, 0, 0, 1]])
B = np.array([[0], [0], [1/tc], [0]])
C = np.array([1, 0, 0, 0])
D = [0]

# 1 VARIANCE, continuous.
Sigma_j   = ca.MX.zeros(A.shape[0],A.shape[1])
Sigma_jp  = ca.MX.zeros(n,1)
for j in np.arange(1,n):
  dSigma_j        = A @ Sigma_j + Sigma_j.T @ A.T + \
                    (u[j] * u[j] * kVar) * (B @ B.T) 
  Sigma_j         = Sigma_j + (dSigma_j)*dt           #Euler
  Sigma_jp[j]   = Sigma_j[0,0]           #sigma position over time. 

# 2. DYNAMICS, as state-space linear system
for j in np.arange(1,n): #julia 2:n
    opti.subject_to(p[j] == p[j-1] + dt*v[j-1])
    opti.subject_to(v[j] == v[j-1] + dt*(F[j-1])) #EOM
    opti.subject_to(F[j] == F[j-1] + dt*1/tc*(u[j-1]-Fdot[j-1]))
    opti.subject_to(Fdot[j] == Fdot[j-1] + (Fddot[j-1]))

#3. OBJECTIVE: ENERGY need to be set here. 
for j in np.arange(0,n):#julia j in 1:n-1
  opti.subject_to(posPowJ[j]  == v[j] * F[j])
  opti.subject_to(slackPosPowJ[j] >=posPowJ[j])
  opti.subject_to(slackPosPowJ[j] >=0)

# 4. SLACK vars for fdotdot
for j in np.arange(0,n-1): # during hold period, keep velocity zero
  opti.subject_to(slackFddp[j] >=Fddot[j])
  opti.subject_to(slackFddp[j] >=0)
  opti.subject_to(slackFddn[j] <=Fddot[j])
  opti.subject_to(slackFddn[j] <=0)

p_f = 1
v_f = 0
# 5. Initial conditions and final conditions
opti.subject_to(p[0] == 0)
opti.subject_to(v[0] == 0)
opti.subject_to(p[ih] == p_f)
opti.subject_to(v[ih] == v_f)  # final condition is at ih

if CONCYCLIC1 == 1:
  opti.subject_to(F[0] == -F[-1])
  opti.subject_to(Fdot[0] == -Fdot[-1])
  opti.subject_to(Fddot[0] == -Fddot[-1])
else:
  opti.subject_to(u[-1] == 0)

# opti.subject_to(Fdot[ih] == 0)

# 6. HOLD period
for j in np.arange(ih,n): # julia ih+1:n # during hold period, keep velocity zero
    opti.subject_to(v[j] == 0)

# OBJECTIVES: E(W+FR) + T
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
opti.set_value(kT,1)
objTimeJ = kT*T

# Objective: minimize Fdot squared over movement period
objVarianceHold = ca.sum1(Sigma_jp[ih:])
opti.minimize(objVarianceHold)

# sigma limit
Sigma_limit = opti.parameter(1,1)
opti.subject_to(objVarianceHold <= Sigma_limit)

opti.set_value(Sigma_limit, 5)
opti.set_initial(T,0.5)

maxIter = 1000
pOpt = {"expand":True}
sOpt = {"max_iter"        : maxIter,
        "constr_viol_tol" : 1e-2,
        "dual_inf_tol"    : 1e-2}
opti.solver('ipopt',pOpt,sOpt)
f,ax = plt.subplots(2,2)
def callbackPlots(i):
    tt = opti.debug.value(T)
    ax[0,0].plot(tauvec*tt,opti.debug.value(p),color=(1,.8-.8*i/(maxIter),1))
    ax[0,1].plot(tauvec*tt,opti.debug.value(v),color=(1,.8-.8*i/(maxIter),1))
    ax[1,0].plot(tauvec*tt,opti.debug.value(Sigma_jp),color=(1,.8-.8*i/(maxIter),1))
    ax[1,1].plot(tauvec[0:-1]*tt,opti.debug.value(Fddot),color=(1,.8-.8*i/(maxIter),1))
opti.callback(callbackPlots)

try:
      sol = opti.solve()
except:
  print("did not find solution")

print("T:")
print(sol.value(T))

print("pos work J")
print(sol.value(objPosWorkJ))
print("fr cost J ")
print(sol.value(objFrJ))
print("time J ")
print(sol.value(objTimeJ))

print("variance Sigma_jp")
print(sol.value(objVarianceHold))

# max(opti.debug.value(opti.debug.value(V11)))
#%%
Topt = sol.value(T)
fig,ax1 = plt.subplots(3,3)

ax1[0,0].plot(tauvec*Topt,sol.value(p))
ax1[0,0].set_title("p")

ax1[0,1].plot(tauvec*Topt,sol.value(v))
ax1[0,1].set_title("v")

ax1[0,2].plot(tauvec*Topt,sol.value(F))
ax1[0,2].set_title("F")

ax1[1,0].plot(tauvec*Topt,sol.value(u))
ax1[1,0].set_title("u")

ax1[1,2].plot(tauvec[0:-1]*Topt,sol.value(slackFddn),\
  tauvec[0:-1]*Topt,sol.value(slackFddp))
ax1[1,2].set_title("slack")

ax1[2,0].plot(tauvec*Topt,sol.value(Fdot))
ax1[2,0].set_title("fdot")

ax1[2,1].plot(tauvec[0:-1]*Topt,sol.value(Fddot))
ax1[2,1].set_title("fddot")

ax1[2,2].plot(tauvec*Topt,sol.value(Sigma_jp))
ax1[2,2].set_title("sigma")



sol.value(objFrJ)
# %%
