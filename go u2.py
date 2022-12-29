# approximate fits for 2023 wong herspiegel kuo for 
# min torque-change + time, and u2 + time. 
# as of 2022-12-20, approach has been to fit approximately 
# the movement duration / speed at 10cm reach by changing
# ku2/kdtor, and then keeping that constant fixed across
# other reaches.
#
# cannot have both, if fit one then don't get other; 
# and peak speed tends to have more of a sqrt shape. 
# 
# empirical data, 10cm reach:
# peakspeed: 0.2 m/s
# duration: 0.8
#%%
import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import plotpresentlib as pp
from matplotlib.figure import Figure
import scipy

sim = reaching.Kinarm()

sim.l1  = 0.3
sim.l2  = 0.29
tval    = 15
sim.l   = np.array([sim.l1,sim.l2])
xystart = np.array([0,0.2])

xyend = xystart+np.array([0,.1])

optipDT   = sim.movementTimeOptSetup(
  theTimeValuation  = tval,
  theN              = 120,   
  theFRCoef         = 8.5e-2,
  thecostFR1U22TR3=3,
  thekdtor=1e-1)

tr,dump = sim.updateGuessAndSolve(optipDT,xystart,xyend,
  theGeneratePlots = 0, theFRCoef=8.5e-2,theTimeValuation = tval)
plt.plot(tr.time,tr.handspeed)
# human: 10cm - 0.2 m/s, 0.8 s. good fit to both
# dtorque: good fit to both with thekdtor1e-1.

# specify the loop
dists    = np.linspace(.01,0.30,15)

# run the loop
traj = list()
duration = np.zeros(dists.shape[0])
peakspeed = np.zeros(dists.shape[0])
for count,d in enumerate(dists):
  xyend = xystart + np.array([0,d])
  tr,dump = sim.updateGuessAndSolve(optipDT, xystart, xyend,
    theGeneratePlots = 0, 
    theTimeValuation = tval)
  traj.append(tr)
  duration[count]   = tr.time[-1]
  peakspeed[count]  = max(tr.handspeed)

#save the results
saveDict = {
    "peakspeed":peakspeed,
    "distance":dists,
    "duration":duration,
    "timeValuation": tval,
    "traj": traj}
scipy.io.savemat('simulationResults/results_dtorque.mat', saveDict)

#plot the results
f,ax = plt.subplots(2,2)
ax[0,0].plot(dists,peakspeed)
ax[0,0].set_ylim([0,1])
ax[1,0].plot(dists,duration)
ax[1,0].set_ylim([0,2])
ax[0,0].set_title("dudt")
for it in traj:
  if it.time.shape[0] == it.handspeed.shape[0]:
    ax[1,1].plot(it.time,it.handspeed)

# %%
## create a U2 optimization
## set the coefficient theku2 to match human data.
## note that for u2 opt, this isn't possible: it's too
## much like minimum mechanical energy. 
## 
## 10cm - 0.2 m/s, 0.8 s.
xyend = xystart + np.array([0,.1])
optipU2   = sim.movementTimeOptSetup(
  theTimeValuation  = tval,
  theN              = 120,   
  thecostFR1U22TR3=2,
  theku2=15)

# specify the loop
dists    = np.linspace(.01,0.30,15)

traju2 = list()
durationu2 = np.zeros(dists.shape[0])
peakspeedu2 = np.zeros(dists.shape[0])
for count,d in enumerate(dists):
  xyend = xystart + np.array([0,d])
  tr,dump = sim.updateGuessAndSolve(optipU2,xystart,xyend,
    theGeneratePlots = 0, theFRCoef=8.5e-2,theTimeValuation = tval)
  traju2.append(tr)
  durationu2[count]   = tr.time[-1]
  peakspeedu2[count]  = max(tr.handspeed)

# save results
saveDict = {
    "peakspeed":peakspeedu2,
    "distance":dists,
    "duration":durationu2,
    "timeValuation": tval,
    "traj":traju2}
scipy.io.savemat('simulationResults/results_u2.mat', saveDict)

# plot results
f,ax2 = plt.subplots(2,2)
ax2[0,0].plot(dists,peakspeedu2)
ax2[0,0].set_ylim([0,1])
ax2[1,0].plot(dists,durationu2)
ax2[1,0].set_ylim([0,2])
for it in traju2:
  if it.time.shape[0] == it.handspeed.shape[0]:
    ax2[1,1].plot(it.time,it.handspeed)
