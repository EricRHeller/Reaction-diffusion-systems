#!/usr/bin/env python
# coding: utf-8

""" SI Fig. 5 """
__author__ = 'Eric Heller'
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.interpolate import make_interp_spline

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
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

# Noise strength
Ω       = 1e+2
Ωarray  = np.logspace(-0.5,+2,1000)
Ωarray *= Ω

#Convert instanton rates between different noise strengths
def convert(ko,S,Oo,On,interface=True):
    if interface==True:
        kn = ko + 0.5 * np.log(On/Oo) + S*(Oo - On)
    else:
        kn = ko + S*(Oo - On)
    return kn

# Import instanton rates
fw     = np.genfromtxt("fw.txt")
bw     = np.genfromtxt("bw.txt")
darray = fw[:,0]
# Import actions
fw_a   = np.genfromtxt("action_fw.txt")
bw_a   = np.genfromtxt("action_bw.txt")

actions_fw = fw_a[:,1]
actions_bw = bw_a[:,1]

# Spline interpolate actions
aspline1 = make_interp_spline(fw_a[:,0], fw_a[:,1], k=3)
aspline2 = make_interp_spline(bw_a[:,0], bw_a[:,1], k=3)

x_smooth = np.linspace(np.min(fw[:,0]), np.max(fw[:,0]), 1001)  # Smooth range for spline fitting
# Evaluate spline curves
a1_smooth = aspline1(x_smooth)
a2_smooth = aspline2(x_smooth)

# Compute instanton rates on the kappa-Omega grid
fw_r = np.zeros((len(x_smooth),len(Ωarray)))
bw_r = np.zeros((len(x_smooth),len(Ωarray)))
fw_pre = np.log(fw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
bw_pre = np.log(bw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
for indo,Oi in enumerate(Ωarray):
    for indr in range(len(actions_fw)):
        if indr < 6:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Oi,interface=True)
        else:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Oi,interface=False) 
    for indr in range(len(actions_bw)):
        if indr < 4:
            bw_pre[indr,indo] = convert(bw_pre[indr,indo],actions_bw[indr],Ω,Oi,interface=True)
        else:
            bw_pre[indr,indo] = convert(bw_pre[indr,indo],actions_bw[indr],Ω,Oi,interface=False) 

# Spline interpolate rates
for indo in range(len(Ωarray)):
    rspline1     = make_interp_spline(fw[:,0], fw_pre[:,indo], k=3)
    r1_smooth    = rspline1(x_smooth)
    fw_r[:,indo] = r1_smooth
    
    rspline2     = make_interp_spline(bw[:,0], bw_pre[:,indo], k=3)
    r2_smooth    = rspline2(x_smooth)
    bw_r[:,indo] = r2_smooth


### Phase diagram as contour plot
fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(9,6))

Z_data = (fw_r < bw_r).astype(int) 
colormap = plt.get_cmap('coolwarm')
cm_blue = mcolors.to_hex(colormap(0.0))
cm_red  = mcolors.to_hex(colormap(1.0))
colors  = [cm_red, cm_blue]
cmap    = mcolors.ListedColormap(colors)

ax1.set_xlabel('$\kappa$', size=24,labelpad=8)
ax1.set_ylabel('$\Omega^{-1}$', size=24, labelpad=12)
ax1.xaxis.set_major_locator(MultipleLocator(5.0))
ax1.xaxis.set_minor_locator(MultipleLocator(2.5))

DM, ΩM = np.meshgrid(x_smooth,1/Ωarray)

contour_plot = ax1.contourf(DM, ΩM, Z_data.T, levels=[0, 0.5, 1], cmap=cmap) # levels define boundaries for colors
cbar = fig.colorbar(contour_plot, ax=ax1, ticks=[0.25, 0.75], format=plt.FuncFormatter(lambda x, pos: r'$\rho_{-}$ stable' if x > 0.5 else r'$\rho_{+}$ stable'))

ax1.set_yscale('log')

plt.xlim(np.min(fw[:,0]),15)

plt.tight_layout()

plt.savefig("1DSchloegl_PhaseDiagram.pdf", format='pdf',dpi=600)
plt.show()




