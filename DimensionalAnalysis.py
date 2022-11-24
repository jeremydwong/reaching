import numpy as np
import casadi as ca
import scipy.io as io 
import os
import scipy.integrate as integ
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import SimpleOpt as so  
import matplotlib.animation as an
import matplotlib.transforms as transforms

from numpy.core.function_base import linspace
from IPython import display
from matplotlib.patches import Ellipse

def timeNorm(g=9.81,l = 1.0):
# def timeNorm(g=9.81,l = 1):
# multiply by this coefficient to nondimensionalize time in s. 
  return np.sqrt(l / g)

def forceNorm(g = 9.81, m = 60.0):
  return 1 / (g * m)

def angularVelocityNorm(g=9.81,l = 1.0):
# def AngularVelocityNorm(g=9.81,l = 1):
# multiply by this coefficient to nondimensionalize angular speed in rad/s
  return (1 / timeNorm(g=g,l=l))

def distanceNorm(l=1.0):
  return 1 / l

def velocityNorm(l = 1.0, g =9.81):
# def timeNorm(g=9.81,l = 1):
# multiply by this coefficient to nondimensionalize translational speed in m/s.
  return 1 / (np.sqrt(l * g))

def powerNorm(l = 1.0, m = 60.0, g = 9.81):
# def timeNorm(g=9.81,l = 1):
# multiply by this coefficient to nondimensionalize work in s.
  return forceNorm(g=g,m=m)*velocityNorm(l=l, g=g)
