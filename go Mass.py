#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 

import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

# Hey Billy, here are the 
loopMasses = [1,2,3,4]
color1 = clr.Color("#e0f3db")
distcolors = list(color1.range_to(clr.Color("#084081"),len(loopMasses)))

sim = reaching.Kinarm()
%config InlineBackend.figure_formats = ['svg']

# time the simulation
tstart = time.time()

trajList = list()
optiList = list()
for i in loopMasses:
  # setup the simulation.
  prevSol = []
  durationFixed = 0.5
  optiPrev = sim.movementTimeOptSetup(
    theN              = 100,
    theDuration = durationFixed,
    theHandMass = i)

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
