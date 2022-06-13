#%%
import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

# loopValuation = [1,2,3]
# color1 = clr.Color("#e0f3db")
# distcolors = list(color1.range_to(clr.Color("#084081"),len(loopValuation)))

sim = reaching.Kinarm()
# %config InlineBackend.figure_formats = ['svg']

tstart = time.time()
prevSol = []
dGuess = 1.0
optiP = sim.movementTimeNoXSetup(
  theTimeValuation  = 1.0, 
  theN              = 200)

xystart = np.array([0,0.2])
traj3, opti3 = sim.updateNoXAndSolve(optiP, 
  xystart, 
  theYBoundary = .3, 
  theDurationInitial = 0.5,
  theTimeValuation = 1,
  theGeneratePlots = 1)

traj4, opti4 = sim.updateNoXAndSolve(optiP, 
  xystart, 
  theYBoundary = .4, 
  theDurationInitial = 0.5,
  theTimeValuation = 1,
  theGeneratePlots = 1)

traj5, opti5 = sim.updateNoXAndSolve(optiP, 
  xystart, 
  theYBoundary = .5, 
  theDurationInitial = 0.5,
  theTimeValuation = 1,
  theGeneratePlots = 1)
#%%
%config InlineBackend.figure_formats = ['svg']
fig,ax = plt.subplots()
ax.plot(traj3.hand[0,:].T,traj3.hand[1,:].T,
  traj4.hand[0,:].T,traj4.hand[1,:].T,
  traj5.hand[0,:].T,traj5.hand[1,:].T)
ax.axis([0.0,.5,0.0,.5])
plt.show()
# %%
