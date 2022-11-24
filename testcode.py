#%%
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

import sys, os

from SimpleOpt import optTrajectories, optiParam
sys.path.append(os.getcwd())
cwd = os.getcwd()
sys.path.append(cwd.rsplit('/', 1)[0])
import ReachingModels as reaching
import plotpresentlib as pp

#0.0560    0.1120     0.1680    0.2240    0.2800
distcolors = pp.bluegreen(4)

sim = reaching.DoublePendulum()
#%config InlineBackend.figure_formats = ['svg']

timeVal = 5
optiP = sim.movementTimeNoXSetup(
  theTimeValuation  = timeVal, 
  theN              = 50,
  gt1lt0            = 1)

trajout  = list()
optout    = list()
for yb in np.array([.1,.2,.3,.4]):
  doplot = 0
  xystart = np.array([0.0,0.1])
  traj, optis = sim.updateNoXAndSolve(optiP, 
    xystart, 
    theYBoundary = yb+xystart[1], 
    theDurationInitial = 1.0,
    theTimeValuation = timeVal,
    theGeneratePlots = doplot)
    
  trajout.append(traj)
  optout.append(optis)

  #%% test pinv
  a = np.array([[1,2,3,4,5,6]]).T
  ainv = np.linalg.pinv(a)
  ainv_check = np.linalg.inv(a.T @ a) @ a.T

# %%
