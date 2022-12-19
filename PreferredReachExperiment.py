#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 
import ReachingModels as reaching
import numpy as np
import matplotlib.pyplot as plt
import colour as clr
import plotpresentlib as pp
from matplotlib.figure import Figure
import scipy

def simulatePreferredReaches(dists = [0.05,.1,0.15,0.2,0.25,0.3],xystart=np.array([0.05,0.2]), 
  timeValuation = 15, n = 60, fr = 8.5e-2, savename = 'resultsTrajectories.mat'):
  # %config InlineBackend.figure_formats = ['svg']
  # model that has equations to compute endpoint jacobians, equations of motion, and energy. 
  sim = reaching.Kinarm()
  sim.l2 = 0.3
  sim.l1 = 0.3
  sim.l = np.array([sim.l1,sim.l2])
  optiPrev = sim.movementTimeOptSetup(
  theTimeValuation  = timeValuation,
  theN              = n,   
  theFRCoef         = fr)
  
  trajList = list()  
  duration = np.zeros([len(dists)])
  peakspeed = np.zeros([len(dists)])
  distance = np.zeros([len(dists)])

  for count,i in enumerate(dists):
    print("i" + str(i))
    xyend = xystart + np.array([0,i])
    temptraj, optiPrev = sim.updateGuessAndSolve(optiPrev,xystart = xystart, xyend = xyend, theTimeValuation=timeValuation,theGeneratePlots=0)
    trajList.append(temptraj)
    duration[count]  = temptraj.time[-1]
    peakspeed[count] = max(temptraj.handspeed)
    distance[count] = i

  saveDict = {
    "traj":trajList,
    "peakspeed":peakspeed,
    "distance":distance,
    "duration":duration}

  scipy.io.savemat('simulationResults/'+savename, saveDict)
  return trajList, sim, saveDict

tvals = np.linspace(101,200,100)
ls = np.linspace(0.01,0.35,35)
duration  = np.zeros([tvals.shape[0],ls.shape[0]])
peakspeed = np.zeros([tvals.shape[0],ls.shape[0]])
distance  = np.zeros([tvals.shape[0],ls.shape[0]])
timeValuation = np.zeros([tvals.shape[0],ls.shape[0]])
  
for count,itval in enumerate(tvals):
  sname = "trajWtval"+str(int(itval))+".mat"
  
  trajs, sim, saveDict = simulatePreferredReaches(dists=ls,
    timeValuation = itval, savename=sname)
  
  # some printing and storing of data
  print("-------------------- completed" + str(int(itval)) + " --------------------")
  print("--------------------------------------------------------------------------------")
  print("--------------------------------------------------------------------------------")  
  distance[count,] = saveDict["distance"]
  duration[count,] = saveDict["duration"]
  peakspeed[count,] = saveDict["peakspeed"]
  timeValuation[count,] = np.ones([ls.shape[0]])*itval
  
  # save data
  saveDict = {
    "peakspeed":peakspeed,
    "distance":distance,
    "duration":duration,
    "timeValuation": timeValuation}
  scipy.io.savemat('simulationResults/results20221216.mat', saveDict)

    


# %%
