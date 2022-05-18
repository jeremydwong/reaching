#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 

import DoublePendulumClass as dp
import numpy as np
import matplotlib.pyplot as plt

sim = dp.PointMass()
#%config InlineBackend.figure_formats = ['svg']
prevSol = []
dGuess = 1.0
optiPrev = sim.movementTimeOptSetup(
  theTimeValuation  = 1.0, 
  theN              = 100)

xystart = np.array([0,0.2])

#%%
import time
tstart = time.time()
trajOrig, opti1 = sim.updateGuessAndSolve(optiPrev, 
  xystart, 
  xystart + np.array([0,.1]), 
  theDurationInitial = 0.5,
  theTimeValuation = 1,
  theGeneratePlots = 1)
tend = time.time()
dur = tend-tstart
print("duration was "+str(dur) +" seconds.")
#cProfile.run('sim.updateGuessAndSolve(optiPrev, xystart, xystart + np.array([0,.1]), theDurationInitial = 0.5, theTimeValuation = 1, theGeneratePlots = 1)')
dGuess = trajOrig.duration
#%%
tstart = time.time()
traj1to2, opti1to2 = sim.updateGuessAndSolve(opti1, xystart, xystart + np.array([0,.11]), 
      theDurationInitial = dGuess,
      theTimeValuation = 1,
      theGeneratePlots = 1)

tend = time.time()
dur2 = tend-tstart
print("duration was "+str(dur2) +" seconds.")
#%%
traj_origto2, opti_origto2 = sim.updateGuessAndSolve(optiPrev, xystart, xystart + np.array([0,.11]), 
      theDurationInitial = dGuess,
      theTimeValuation = 1,
      theGeneratePlots = 1)
dGuess = trajOrig.duration


#%%
energy = sim.energy(trajOrig.Q,trajOrig.QDot,trajOrig.U,trajOrig.time)
plt.plot(trajOrig.time, energy.e_k,
  trajOrig.time,energy.e_mech)

# %%
