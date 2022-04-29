#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 
import DoublePendulumClass as dp
import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_formats = ['svg']

# model that has equations to compute endpoint jacobians, equations of motion, and energy. 
sim = dp.kinarmModel()

x = 0.0
ys = 0.2
xystart = np.array([x,ys])

loopValuation = [5,3,1,.75,.5,.25]
loopValuation = [1,3]
loopdist = np.array([.15])

distcolors = ['#f7fcf0','#e0f3db','#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']

M = len(loopdist)
N = len(loopValuation)
duration = np.zeros([M, N])
J = np.zeros([M, N])
frCost = np.zeros([M, N])
workCost = np.zeros([M, N])
timeCost = np.zeros([M, N])
peakhandspeed = np.zeros([M, N])

# get all trajectories
trajRows = list()
trajCols = list() 

# get all optimization results
optiRows = list()
optiCols = list()

mDist = 0
nVal = 0
solution = []
for i in loopdist:
  for j in loopValuation:
    solution, \
      duration[mDist,nVal], J[mDist,nVal], workCost[mDist,nVal], frCost[mDist,nVal], timeCost[mDist,nVal], peakhandspeed[mDist,nVal],trajResult = \
      sim.movementTimeOpt(xystart,xystart + np.array([x,i]), \
      theTimeValuation=j,theN = 100, theGeneratePlots=0,sol=solution, LINEAR_GUESS=False, theDurationGuess = 1.0)
    nVal += 1
    trajCols.append(trajResult)
    optiCols.append(solution)
  nVal += 1
  mDist = 0
  trajRows.append(trajCols)
  optiRows.append(optiCols)
  trajCols = []
# %%


#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 
import DoublePendulumClass as dp
import numpy as np
import matplotlib.pyplot as plt

#%config InlineBackend.figure_formats = ['svg']
xfixed = 0.0
ystart = 0.2
dy = 0.1
xystart = np.array([xfixed,ystart])

kinarm = dp.kinarmModel()
optiKin, duration, J, workCost, frCost, timeCost,\
  peakhandspeed,resultKin = \
  kinarm.movementTimeOpt(xystart,xystart + np.array([xfixed,dy]), \
    theTimeValuation=3.0,theN = 100, theGeneratePlots=1)

energy = kinarm.energy(resultKin.Q,resultKin.QDot,resultKin.U,resultKin.time)
fig,ax = plt.subplots()
plt.plot(resultKin.time,energy.e_g+energy.e_k,\
  resultKin.time,energy.e_mech)
#%% 
#%%
