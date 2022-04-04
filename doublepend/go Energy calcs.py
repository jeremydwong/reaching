
#%% Loop across distances and valuations to generate double-pendulum distance/time predictions. 
import DoublePendulumClass as dp
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']

sim = dp.doublePendulum()

xfixed = 0.0
ystart = 0.1
xystart = np.array([xfixed,ystart])

jco = 0
ico = 0
for j in [1]:
  for dy in [.1]:
    duration, J, workCost, frCost, timeCost,\
      peakhandspeed,result = \
      sim.movementTimeOpt(xystart,xystart + np.array([xfixed,dy]), \
        theTimeValuation=j,theN = 100, theGeneratePlots=1)

energy = sim.energy(result.Q,result.QDot,result.U,result.time)
#%%
f,ax = plt.subplots()
ax.plot(result.time, energy.e_k[0,:])
ax.plot(result.time, energy.e_mech[0,:])
plt.show()

# %%
