from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import colour as clr

def figsizeInches(key="fourth"):
  heightKeynoteCM = 38
  widthKeynoteCM = 68
  cmPerIn = 2.54
  dimSlideCM = np.array([widthKeynoteCM,heightKeynoteCM])
  switcher = {
    "fourth": dimSlideCM*1/4,
    "third": dimSlideCM*1/3,
    "squarefourthheight": np.array([dimSlideCM[1],dimSlideCM[1]]) * 1/4,
    }
  return switcher.get(key, dimSlideCM*1/4) / cmPerIn

# def set_size(w,h, ax=None):
# set size in inches. helper function called from figurefyTalk()
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: 
      ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def figurefyTalk(ax = None):
  if not ax: 
    ax=plt.gca()
  lfont = {'fontname':'Helvetica'}
  widthHeight = figsizeInches("fourth")
  set_size(widthHeight[0],widthHeight[1])
  plt.xticks(fontsize = 20, **lfont)
  plt.yticks(fontsize = 20, **lfont)
  plt.xlabel(ax.get_xlabel(),fontsize = 28,**lfont)
  plt.ylabel(ax.get_ylabel(),fontsize = 28,**lfont)
  
def figurefyMatch(ax = None):
  if not ax: 
    ax=plt.gca()
  lfont = {'fontname':'Helvetica'}
  #fontlabel size: 26.93. 
  #fonttick size: 16.16.
  cmPerIn = 2.54   
  widthHeight = np.array([17,9.5])/cmPerIn #Axis dimensions in cm: [W, H]: [17.00 , 9.50]. 
  set_size(widthHeight[0],widthHeight[1])
  plt.xticks(fontsize = 16.16, **lfont)
  plt.yticks(fontsize = 16.16, **lfont)
  plt.xlabel(ax.get_xlabel(),fontsize = 26.93)
  plt.ylabel(ax.get_ylabel(),fontsize = 26.93)

def bluegreen(n):
  color1 = clr.Color("#e0f3db")
  return list(color1.range_to(clr.Color("#084081"),n))
