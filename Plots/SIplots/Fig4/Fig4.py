#!/usr/bin/env python
# coding: utf-8

""" SI Fig 4 """                                                 
__author__ = 'Eric Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

# Plotting options
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
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

# RP-Parameters
N       = 512
# Physical parameters
dof     = 40

# Kappa=8, FW
minRFW  = np.genfromtxt("minR_FW.txt")
minPFW  = np.genfromtxt("minP_FW.txt")
instxFW = np.genfromtxt("xinitFW.txt").reshape(-1,dof)
ionexFW = np.concatenate([instxFW.reshape(-1,dof),minPFW.reshape(1,dof)],axis=0)
instpFW = np.concatenate([minRFW.reshape(1,dof),ionexFW],axis=0)

# Kappa=8, BW
minRBW  = np.genfromtxt("minR_BW.txt")
minPBW  = np.genfromtxt("minP_BW.txt")
instxBW = np.genfromtxt("xinitBW.txt").reshape(-1,dof)
ionexBW = np.concatenate([instxBW.reshape(-1,dof),minPBW.reshape(1,dof)],axis=0)
instpBW = np.concatenate([minRBW.reshape(1,dof),ionexBW],axis=0)

T       = 60
tau     = T / N

### Plotting ###
fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17,6), gridspec_kw={'width_ratios': [0.81, 1]} )

colormap = plt.get_cmap("coolwarm") 
####
#Plot
tplot      = np.linspace(0.0,T,N+1)
xplot      = np.linspace(0.0,dof-1,dof)
TIME, X    = np.meshgrid(tplot, xplot)

# --- force normalization and levels regardless of data ---
vmin, vmax = 0.5, 1.6
norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)
levels     = np.linspace(vmin, vmax, 21)

cmFW       = ax1.contourf(TIME, X, instpFW.T, levels=levels, norm=norm, extend='both', cmap=colormap)
cmBW       = ax2.contourf(TIME, X, instpBW.T, levels=levels, norm=norm, extend='both', cmap=colormap)


# --- force a full-range colorbar ---
cbar = fig.colorbar(cmFW, ax=ax2,ticks=np.linspace(vmin, vmax, 3))
cbar.ax.tick_params(labelsize=24)
cbar.set_label(r"$\rho(x,t)$", fontsize=24, labelpad=12)

# --- axes styling ---
ax1.set_xlabel("$t$", size=24, labelpad=8)
ax2.set_xlabel("$t$", size=24, labelpad=8)

ax1.set_ylabel("$x$", size=24, labelpad=12)
ax1.xaxis.set_major_locator(MultipleLocator(20))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.yaxis.set_minor_locator(MultipleLocator(5))

ax2.xaxis.set_major_locator(MultipleLocator(20))
ax2.xaxis.set_minor_locator(MultipleLocator(10))
ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.yaxis.set_minor_locator(MultipleLocator(5))

ax1.set_xlim(np.min(tplot), np.max(tplot))
ax1.set_ylim(np.min(xplot), np.max(xplot))

ax2.set_xlim(np.min(tplot), np.max(tplot))
ax2.set_ylim(np.min(xplot), np.max(xplot))

ax2.set_yticklabels([])

plt.tight_layout()
fig.subplots_adjust(wspace=0.2)


plt.savefig("1DSchloegl_Paths.pdf",format='pdf',dpi=600)

plt.show()




