import casadi as ca
import SimpleOpt as so

opti = ca.Opti()
x = opti.variable(self.DoF*4, theN+1) 

# Constants
p_0 = 0    # Initial position
v_0 = 0    # Initial velocity
p_f = 1    # Final position
v_f = 0    # Final velocity (at step nm)

def trapInt(t,inVec)
  sumval = 0.
  for ii in 1:length(t)-1
      sumval = sumval + (inVec[ii]+inVec[ii+1])/2.0 * (t[ii+1]-t[ii])
  end
  return sumval
end

# Eye movements
n = 100 # total of 100 ms
nm = 50 # finish at 50 ms
Δt = 1/n # in s
Bs  = 0
M   = 1
eyefrresults = []

nabs(x) = x>=0 ? x : -0.2*x


  # Start with continuous-time state space, then discrete time
  A = [0 1 0 0; 0 -Bs/M 1/M 0; 0 0 0 1; 0 0 0 0] # position, velocity, F, Fdot just with damping
  B = [0 0 0 1]'
  C = [1 0 0 0]
  D = [0]
  sseye = ss(A, B, C, D)
  ssdeye = c2d(sseye, Δt)[1,1] # exp(A Δt)

  # Create JuMP model, using Ipopt as the solver
  eyefr = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

  @variables(eyefr, begin  # State variables
      p[1:n]                # Position
      v[1:n]                # Velocity
      F[1:n]                # Force
      Fdot[1:n]             # Fdot
      Fddot[1:n-1]          # Fddot control
      V11[1:n-1]            # variance
      V12[1:n-1]            # variance
      V22[1:n-1]            # variance
      Ek[1:n-1]             # energy  
      slackFddp[1:n-1]
      slackFddn[1:n-1]
      slackPosPowJ[1:n-1]
      posPowJ[1:n-1]
  end)
  T = 1.0

  # 1. Initial conditions and final conditions
  @constraints(eyefr, begin
      p[1] == 0
      v[1] == 0
      p[nm] == p_f
      v[nm] == v_f  # final condition is at nm
      F[1] == 0
      Fdot[1] == 0
      F[nm] == 0
      Fdot[nm] == 0
  end)

  # 2. hold period
  for j in nm+1:n # during hold period, keep velocity zero
      @constraint(eyefr, v[j] == 0)
  end

  # 3. slack vars for fdotdot
  for j in 1:n-1 # during hold period, keep velocity zero
    @constraint(eyefr, slackFddp[j] >=Fddot[j])
    @constraint(eyefr, slackFddp[j] >=0)
    @constraint(eyefr, slackFddn[j] <=Fddot[j])
    @constraint(eyefr, slackFddn[j] <=0)

  end
  
  # 4. Dynamics, as state-space linear system
  for j in 2:n
      @constraint(eyefr, p[j] == p[j-1] + Δt*v[j-1])
      @constraint(eyefr, v[j] == v[j-1] + Δt*(-Bs*v[j-1] + F[j-1])) #EOM
      @constraint(eyefr, F[j] == F[j-1] + Δt*Fdot[j-1])
      @constraint(eyefr, Fdot[j] == Fdot[j-1] + Δt*Fddot[j-1])
  end

  # 5. variance
  V11expr = Matrix{GenericQuadExpr}(undef,n-1,n-1)
  #V12expr = Matrix{GenericQuadExpr}(undef,n-1,n-1)
  #V22expr = Matrix{GenericQuadExpr}(undef,n-1,n-1)
  # For signal-dependent noise
  for myt in 2:n # changed to n-1
    # inner summation
    for i in 1:myt-1
      dimMat = [0 0 0 0; 0 T 0 0; 0 0 T 0; 0 0 0 T]
      AB = (ssdeye.A*dimMat)^(myt-i-1)*(dimMat * ssdeye.B)
      ABBA = AB*AB' 
      V11expr[myt-1,i] = ABBA[1,1]*F[i]*F[i]
    end
  end

  #6. all expressions for variance, energy need to be set here. 
  for j in 1:n-1
    @constraint(eyefr, V11[j] == sum(sum(V11expr[j,1:j],dims=1)))   # setting covariance11
    @constraint(eyefr, posPowJ[j]  == v[j] .* F[j] * T)
    @constraint(eyefr, slackPosPowJ[j] >=posPowJ[j])
    @constraint(eyefr, slackPosPowJ[j] >=0)
  end
  
  # Objectives
  kFR     = 1e-2
  objFrJ  = kFR * sum(slackFddn + slackFddp)
  
  tvec    = range(0,1,length=100)*T

  kPosWorkW  = 4.2
  objPosWorkJ = kPosWorkW * trapInt(tvec[1:end-1],slackPosPowJ)

  kT      = 1
  objTimeJ= kT*T

  #minimize Fdot squared over movement period
  @objective(eyefr, Min, sum(slackFddp - slackFddn))
  #@constraint(eyefr, )

  # Solve for the control and state
  println("Solving...")
  status = optimize!(eyefr)

  push!(eyefrresults, (t=(1:n)*Δt,p=value.(p),v=value.(v),F=value.(F),Fddot=value.(Fddot)))

p1 = plot(eyefrresults[1].t, [res.p for res in eyefrresults], xlabel="t", ylabel="position", title="Force-rate damping")
p2 = plot(eyefrresults[1].t, [res.v for res in eyefrresults], xlabel="t", ylabel="velocity")
p3 = plot(eyefrresults[1].t, [res.F for res in eyefrresults], xlabel="t", ylabel="F")
p4 = plot(eyefrresults[1].t[1:n-1], [res.Fddot[1:n-1] for res in eyefrresults], xlabel="t", ylabel="Fddot")
plot(p1,p2,p3,p4,layout=(2,2),label="")


oP   = optiParam(opti, N = theN) # we attach all symbolic opt variables to oP, to be returned.
