#%%
import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

# loopValuation = [1,2,3]
# color1 = clr.Color("#e0f3db")
# distcolors = list(color1.range_to(clr.Color("#084081"),len(loopValuation)))

sim = reaching.DoublePendulum()
%config InlineBackend.figure_formats = ['svg']

tstart = time.time()
prevSol = []
dGuess = 1.0
optiP = sim.movementSetup(
  theTimeValuation  = 1.0, 
  theN              = 200)

xystart = np.array([-0.2,0.2])
trajChase, optiChase = sim.updateChaseAndSolve(optiP, 
  xystart, 
  yoffset = 0.3,
  yspeed = 0.05,
  theDurationGuess=.3,
  theTimeValuation = 1,
  theGeneratePlots = 1)
# %%
