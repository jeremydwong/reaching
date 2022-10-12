#%%
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time

import sys, os
sys.path.append(os.getcwd())
cwd = os.getcwd()
sys.path.append(cwd.rsplit('/', 1)[0])
import ReachingModels as reaching
import plotpresentlib as pp

#0.0560    0.1120     0.1680    0.2240    0.2800
distcolors = pp.bluegreen(5)

sim = reaching.DoublePendulum()
#%config InlineBackend.figure_formats = ['svg']

optiP = sim.movementTimeNoXSetup(
  theTimeValuation  = 1.0, 
  theN              = 200)

# %%
