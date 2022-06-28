#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 

import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

loopValuation = [1,2,3]
color1 = clr.Color("#e0f3db")
distcolors = list(color1.range_to(clr.Color("#084081"),len(loopValuation)))

sim = reaching.DoublePendulum()
#%config InlineBackend.figure_formats = ['svg']

tstart = time.time()
prevSol = []
dGuess = 1.0
optiLow = sim.movementTimeOptSetup(
  theTimeValuation  = 1.0, 
  theN              = 20,
  discreteOrCont='continuous')  

xystart = np.array([-.10,0.2])

trajOrig, opti1 = sim.updateGuessAndSolve(optiLow, 
  xystart, 
  xystart + np.array([0,.1]), 
  theTimeValuation = 1,
  theGeneratePlots = 1)

tstart = time.time()
prevSol = []
dGuess = 1.0
optiHigh = sim.movementTimeOptSetup(
  theTimeValuation  = 1.0, 
  theN              = 100,
  discreteOrCont='continuous')  


a,b=sim.interpolateGuessAndSolve(opti1,optiHigh)

#%%
tend = time.time()
durIncludingSetup = tend-tstart

tstart = time.time()
trajOrig, opti1 = sim.updateGuessAndSolve(optiPrev, 
  xystart, 
  xystart + np.array([0,.1]), 
  theDurationGuess = 0.5,
  theTimeValuation = 1,
  theGeneratePlots = 1)
tend = time.time()
durUpdate = tend-tstart
print("sim time was "+str(durUpdate) +" seconds.")
#cProfile.run('sim.updateGuessAndSolve(optiPrev, xystart, xystart + np.array([0,.1]), theDurationInitial = 0.5, theTimeValuation = 1, theGeneratePlots = 1)')
dGuess = trajOrig.duration

tstart = time.time()
traj1to2, opti1to2 = sim.updateGuessAndSolve(opti1, xystart, xystart + np.array([0,.11]), 
      theDurationGuess = dGuess,
      theTimeValuation = 1,
      theGeneratePlots = 1)


#%%
tend = time.time()
durWarmStart = tend-tstart
print("duration was "+str(durWarmStart) +" seconds.")

traj_origto2, opti_origto2 = sim.updateGuessAndSolve(optiPrev, xystart, xystart + np.array([0,.11]), 
      theDurationGuess = dGuess,
      theTimeValuation = 1,
      theGeneratePlots = 1)
dGuess = trajOrig.duration

energy = sim.energy(trajOrig.Q,trajOrig.QDot,trajOrig.U,trajOrig.time)
fig,ax = plt.subplots()
ax.plot(trajOrig.time, energy.e_k,label = "kinetic energy")
ax.plot(trajOrig.time,energy.e_mech, label = "mechanical work")
ax.set_ylabel("Energy (J)")
ax.set_ylabel("Time (s)")
# %%
