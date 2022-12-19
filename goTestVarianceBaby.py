# %% [markdown]
# ## Force-rate for double-integrator eye with hold
# The hold period makes it trickier to optimize the movement. It's quite a bit easier not to have the hold. We don't really get bell shape, need to modify final conditions.
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
  
def matpow(inmat,n):
  inmatp = inmat
  for j in range(0,n-1):
    inmatp=inmatp @ inmat
  return inmatp

opti = ca.Opti()
T = opti.variable(1,1)
opti.subject_to(T==.3)

n     = 200 # total of 100 supports
nm    = n-20  # finish & hold at index nm
dtau  = 1/n # dimensionless dtau
Bs    = 0   # damping
M     = 1   # Mass
kVar  = 1    # coefficient of variation from of positional variance from u^2.

p = opti.variable(n,1)
v = opti.variable(n,1) 
F = opti.variable(n,1) 
Fdot = opti.variable(n,1) 
Fddot = opti.variable(n-1,1) 
Sigma_jp = opti.variable(n-1,1) 
Ek = opti.variable(n-1,1) 
slackFddp = opti.variable(n-1,1) 
slackFddn = opti.variable(n-1,1) 
posPowJ = opti.variable(n,1) 
slackPosPowJ = opti.variable(n,1) 
V11expr = opti.variable(n-1,n-1)

# Constants
p_0 = 0    # Initial position
v_0 = 0    # Initial velocity
p_f = 1    # Final position
v_f = 0    # Final velocity (at step nm)
T = 1

# Eye movements
n = 100 # total of 100 ms
nm = n-1 # finish at 50 ms

Δt = T/n # in s
kVar = 1
Bs = 0
M = 1
eyefrresults = []

# Start with continuous-time state space, then discrete time
A = [0 1 0 0; 0 -Bs/M 1/M 0; 0 0 0 1; 0 0 0 0] # position, velocity, F, Fdot just with damping
B = [0 0 0 1]
C = [1 0 0 0]
D = [0]
sseye = ctrl.ss(A, B, C, D)
ssdeye = ctrl.c2d(sseye, Δt)[1,1] # exp(A Δt)

# Create JuMP model, using Ipopt as the solver
# eyefr = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

#register(eyesdn, :nabs, 1, nabs, autodiff=true)

# Variance, signal-dependent noise, Discrete-time
for myt in 1:n # changed to n-1
  # inner summation
  for i in 1:myt
    AB = matpow(ssdeye.A,(myt-i-1)) @ ssdeye.B  
    ABBA = AB @ AB' 
    V11expr[myt-1,i] = ABBA[1,1]*F[i]*F[i]
    V12expr[myt,i] = ABBA[1,2]*F[i]*F[i]
    V22expr[myt,i] = ABBA[2,2]*F[i]*F[i]
  end
end

# Variance, signal-dependent noise, continuous.

Sigma_j = zeros(size(sseye.A,1),size(sseye.A,2))
Sigma_jpexpr   = Matrix{GenericQuadExpr}(undef,n-1,1)
for j in 1:n-1
  #dΣⱼ = sseye.A * Σⱼ + Σⱼ' * sseye.A' + sseye.B*u[j]*u[j]*sseye.B'
  dSigma_j          = sseye.A * Sigma_j + Sigma_j' * sseye.A' + 
                    (Fddot[j] * Fddot[j] * kVar) * (sseye.B * sseye.B') 
  global Sigma_j           = Sigma_j + (dSigma_j)*Δt           #Euler
  Sigma_jpexpr[j]   = Sigma_j[1,1]
end

# Dynamics, EULER
for j in 2:n
  @constraint(eyefr, p[j] == p[j-1] + Δt*v[j-1])
  @constraint(eyefr, v[j] == v[j-1] + Δt*F[j-1])
  @constraint(eyefr, F[j] == F[j-1] + Δt*Fdot[j-1])
  @constraint(eyefr, Fdot[j] == Fdot[j-1] + Δt*Fddot[j-1])
  @constraint(eyefr, V11[j] == sum(sum(kVar * V11expr[j,1:j-1],dims=1)))
  @constraint(eyefr, V12[j] == sum(sum(kVar * V12expr[j,1:j-1],dims=1)))
  @constraint(eyefr, V22[j] == sum(sum(kVar * V22expr[j,1:j-1],dims=1)))
  @constraint(eyefr, Sigma_jp[j-1] == Sigma_jpexpr[j-1])
end


# Initial conditions and final conditions
@constraints(eyefr, begin
    p[1] == 0
    v[1] == 0
    p[nm] == p_f
    v[nm] == v_f  # final condition is at nm
    
    
    F[1] == 0
    F[nm] == 0
    #Fdot[1] == 0
    #Fdot[nm] == 0
end)

for j in nm+1:n # during hold period, keep velocity zero
    @constraint(eyefr, v[j] == 0)
end

 objVarHold = sum(V11[nm+1:end])
 objVarHoldCont = sum(Sigma_jp[nm+1:end])

# Objective: minimize Fdot squared over movement period
# @objective(eyefr, Min, sum(Fddot[1:n-1].^2))
@objective(eyefr, Min, objVarHoldCont)

# Solve for the control and state
println("Solving...")
status = optimize!(eyefr)

push!(eyefrresults, (t=(1:n)*Δt,p=value.(p),v=value.(v),F=value.(F),Fddot=value.(Fddot),variDisc=value.(V11),variCont=value.(Sigma_jp)))

using Plots
p1 = plot(eyefrresults[1].t, [res.p for res in eyefrresults], xlabel="t", ylabel="position", title="Variance Minimization")
p2 = plot(eyefrresults[1].t, [res.v for res in eyefrresults], xlabel="t", ylabel="velocity")
p3 = plot(eyefrresults[1].t, [res.F for res in eyefrresults], xlabel="t", ylabel="F")
p4 = plot(eyefrresults[1].t[1:end-1], [res.variCont for res in eyefrresults], xlabel="t", ylabel="variance")
plot(p1,p2,p3,p4,layout=(2,2),label="")

# %% [markdown]
# ## Minimize work + force-rate
# Work alone promotes coasting, but force-rate is relative insensitive to it.

# %%