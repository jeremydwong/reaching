#%%
import DoublePendulumClass as dp
import numpy as np

sim = dp.doublePendulum()

x = 0.0
ys = 0.1
xystart = np.array([x,ys])

loopdist = np.array([.05, .1,.2,.3,.4,.5])
loopValuation = np.array([.05,.5,.75,1,2])
duration = np.ndarray([len(loopValuation), len(loopdist)])
J = np.ndarray([len(loopValuation), len(loopdist)])
frCost = np.ndarray([len(loopValuation), len(loopdist)])
workCost = np.ndarray([len(loopValuation), len(loopdist)])
timeCost = np.ndarray([len(loopValuation), len(loopdist)])
results = list()

jco = 0
ico = 0
for j in loopValuation:
  for i in loopdist:
    duration[jco,ico],J[jco,ico],workCost[jco,ico],frCost[jco,ico],timeCost[jco,ico],result = \
      sim.movementTimeOpt(xystart,xystart + np.array([x,i]), \
        theTimeValuation=j, N = 30)
    ico += 1
    results.append(result)
  jco += 1
  ico = 0
# %%
import matplotlib.pyplot as plt
for v in range(0,len(loopValuation)):
  f,ax = plt.subplots()
  plt.plot(loopdist,duration[v,:])
# %%
