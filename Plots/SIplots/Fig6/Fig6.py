#!/usr/bin/env python
# coding: utf-8

""" SI Fig. 6 """                                                 
__author__ = 'Eric Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# Plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['axes.linewidth'] = 3.0
matplotlib.rcParams['xtick.major.size'] = 15
matplotlib.rcParams['xtick.minor.size'] = 12
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.width'] = 2
matplotlib.rcParams['xtick.direction'] = "inout"
matplotlib.rcParams['ytick.major.size'] = 15
matplotlib.rcParams['ytick.minor.size'] = 12
matplotlib.rcParams['xtick.labelsize'] = 24
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.width'] = 2
matplotlib.rcParams['ytick.direction'] = "inout"
matplotlib.rcParams['legend.fontsize'] = 18
#matplotlib.rcParams['legend.loc'] = "lower left"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

colormap = plt.get_cmap('coolwarm')
cm_blue = mcolors.to_hex(colormap(0.0))
cm_red  = mcolors.to_hex(colormap(1.0))

# Import fixed-point structures
minR = np.genfromtxt("minR.txt")
minP = np.genfromtxt("minP.txt")
TS   = np.genfromtxt("TS.txt")


fig,ax = plt.subplots(figsize=(8.5,6))

# Total lipid density
ρ      = 1
labels = ['PIP$_1$','PIP$_2$','E$_1^{+}$','E$_2^+$','E$_1^-$']
xlabels = np.arange(0,5,1)

# Add PIP1 via constraint
reactant = np.append(np.array([ρ-minR[0]]),np.array([minR[0]]))
reactant = np.append(reactant,minR[1:])
product = np.append(np.array([ρ-minP[0]]),np.array([minP[0]]))
product = np.append(product,minP[1:])
trans = np.append(np.array([TS[0]]),np.array([ρ-TS[0]]))
trans = np.append(trans,TS[1:])

ax.bar(xlabels-0.2, reactant, width=0.2,                    label='PIP$_1$ dominated'  ,color=cm_blue)
ax.bar(xlabels    , trans   , width=0.2, tick_label=labels, label='Transition state'   ,color='C2')
ax.bar(xlabels+0.2, product , width=0.2,                    label='PIP$_2$ dominated'  ,color=cm_red)

plt.ylabel("Concentration",fontsize=24,labelpad=12)


ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

plt.legend()
plt.tight_layout()
plt.savefig("FixedPoints.pdf", format='pdf',dpi=600)
plt.show()


