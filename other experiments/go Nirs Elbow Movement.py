#%% Billy Herspiegel experiment
# idea: to quantify work and force-rate coefficients where both mechanical work and force-rate are varying. 
# unlike wong cluff kuo 2021, work coefficient is not held fixed. 

import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

# Hey Billy, here are the 
loopDamping = [0,0.5,1,1.5,2]
color1 = clr.Color("#e0f3db")
distcolors = list(color1.range_to(clr.Color("#084081"),len(loopDamping)))

sim = reaching.Kinarm()
%config InlineBackend.figure_formats = ['svg']

# time the simulation
tstart = time.time()

trajList = list()
optiList = list()
for i in loopDamping:
  # setup the simulation.
  prevSol = []
  durationFixed = 0.5
  optiPrev = sim.movementTimeOptSetup(
    theN              = 100,
    theDuration = durationFixed,
    theHandMass = i)

  sim.constrainShoulder()
  
  # pick the start positions in x and y. 
  xystart = np.array([.2,0.2])

  # solve the simulation. 
  trajTemp, optiTemp = sim.updateGuessAndSolve(
    optiPrev,
    xystart,
    xystart + np.array([0,.1]),
    theGeneratePlots    = 1)

  trajList.append(trajTemp)
  optiList.append(optiTemp)

# %%
