#%%
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import time
import sys, os
sys.path.append('/Users/jeremy/Git/optreach/')
import ReachingModels as reaching
import plotpresentlib as pp

sim = reaching.DoublePendulum()
sim.g = 9.81
sim.l[1] = 0.4

optiP = sim.movementTimeOptSetup(
  theTimeValuation  = 1.0, 
  theN              = 100)

xyatkeson3 = np.array([0.53, -0.3694])
xyatkeson7 = np.array([0.5600, 0.2462])
#xystart = np.array([0.53, -0.3694])
up,b1=sim.updateGuessAndSolve(oP = optiP, 
  xystart = xyatkeson3, xyend = xyatkeson7,
  theGeneratePlots=1)

#xystart = np.array([0.53, -0.3694])
down,b2=sim.updateGuessAndSolve(oP = optiP, 
  xystart = xyatkeson7, xyend = xyatkeson3,
  theGeneratePlots=1)

fig,axs = plt.subplots(2,1)
axs[0].plot(up.hand[0,:],up.hand[1,:],
  down.hand[0,:],down.hand[1,:])
axs[0].set_xlim([0,.8])
axs[0].set_ylim([-.4,.4])
axs[1].set_xlabel("horizontal position (m)")
axs[1].set_ylabel("vertical position (m)")

axs[1].plot(up.time,up.handspeed,
  down.time,down.handspeed)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Speed (m/s)")

# %%
optiP = sim.movementTimeOptSetup(
  theTimeValuation  = 1.0, 
  theN              = 100)

berretup = sim.joints2Endpoint([np.pi/2,np.pi/2])
berretdown = sim.joints2Endpoint([-np.pi/2,-np.pi/2])
berretstart = sim.joints2Endpoint([0,0])

bup,bu=sim.updateGuessAndSolve(oP = optiP, 
  xystart = berretstart, xyend = berretup,
  theGeneratePlots=1)

bdown,bd=sim.updateGuessAndSolve(oP = optiP, 
  xystart = berretstart, xyend = berretdown,
  theGeneratePlots=1)

fig,axs = plt.subplots(2,1)
axs[0].plot(bup.hand[0,:],bup.hand[1,:],
  bdown.hand[0,:],bdown.hand[1,:])
axs[0].set_xlim([-.7,.7])
axs[0].set_ylim([-.7,.7])
axs[0].set_xlabel("horizontal position (m)")
axs[0].set_ylabel("vertical position (m)")

axs[1].plot(bup.time,bup.handspeed,
  bdown.time,bdown.handspeed)
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Speed (m/s)")

# %%
