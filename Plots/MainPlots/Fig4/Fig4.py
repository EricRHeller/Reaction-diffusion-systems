#!/usr/bin/env python
# coding: utf-8

""" Main text Fig. 4 """
__author__ = 'Eric Heller'
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline

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
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['xtick.major.pad'] = '10'
matplotlib.rcParams['ytick.major.pad'] = '10'


# Noise strengths
Ω       = 5.56
Ωarray  = np.logspace(0.5,2.5,1000)
Ωarray *= Ω

#Convert instanton rates between different noise strengths
def convert(ko,S,Oo,On,interface='droplet'):
    if interface=='droplet':
        kn = ko +       np.log(On/Oo) + S*(Oo - On)
    elif interface=='stripe':
        kn = ko + 0.5 * np.log(On/Oo) + S*(Oo - On)
    else:
        kn = ko                       + S*(Oo - On)
    return kn


### Import data files
# Rates
fw  = np.genfromtxt("fw.txt")
bw  = np.genfromtxt("bw.txt")
# Actions matching rates
fw_a  = np.genfromtxt("action_fw.txt")
bw_a  = np.genfromtxt("action_bw.txt")
# All actions
fw_all  = np.genfromtxt("action_fw_full.txt")
bw_all  = np.genfromtxt("action_bw_full.txt")

actions_fw = fw_a[:,1]
actions_bw = bw_a[:,1]
# Spline interpolate actions
aspline1 = make_interp_spline(fw_all[:,0], fw_all[:,1], k=3)
aspline2 = make_interp_spline(bw_all[:,0], bw_all[:,1], k=3)
#
x_smooth = np.linspace(np.min(fw[:,0]), np.max(fw[:,0]), 1001)  # Smooth range for spline fitting
# Evaluate spline curves
a1_smooth = aspline1(x_smooth)
a2_smooth = aspline2(x_smooth)

# Compute instanton rates on the kappa-Omega grid
fw_r = np.zeros((len(x_smooth),len(Ωarray)))
bw_r = np.zeros((len(x_smooth),len(Ωarray)))
fw_pre = np.log(fw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
bw_pre = np.log(bw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
for indo,Ωi in enumerate(Ωarray):
    for indr in range(len(actions_fw)):
        if indr < 3:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Ωi,interface='stripe')
        else:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Ωi,interface=False) 
    for indr in range(len(actions_bw)):
        if indr < 5:
            bw_pre[indr,indo] = convert(bw_pre[indr,indo],actions_bw[indr],Ω,Ωi,interface='stripe')
        else:
            bw_pre[indr,indo] = convert(bw_pre[indr,indo],actions_bw[indr],Ω,Ωi,interface=False) 

# Spline interpolate rates
for indo,Ωi in enumerate(Ωarray):
    rspline1     = make_interp_spline(fw[:,0], fw_pre[:,indo], k=3)
    r1_smooth    = rspline1(x_smooth)
    fw_r[:,indo] = r1_smooth
    
    rspline2     = make_interp_spline(bw[:,0], bw_pre[:,indo], k=3)
    r2_smooth    = rspline2(x_smooth)
    bw_r[:,indo] = r2_smooth

######## Paths ########
# RP-Parameters
N           = 2000
# System parameters
species     = 5                   # Number of species
Nreac       = 11                  # Number of reactions
dofx        = 8                   # Number of voxels in x
dofy        = 8                   # Number of voxels in x
voxels = L  = dofx * dofy
dof         = species*voxels - 1  # Degrees of freedom (minus constraint)
ρ           = L                   

# FW
minRFW  = np.genfromtxt("minR_FW.txt")
minPFW  = np.genfromtxt("minP_FW.txt")
instxFW = np.genfromtxt("xinitFW.txt").reshape(-1,dof)
instoFW = np.concatenate([instxFW.reshape(-1,dof),minPFW.reshape(1,dof)],axis=0)
instpFW = np.concatenate([minRFW.reshape(1,dof),instoFW],axis=0)

# BW
minRBW  = np.genfromtxt("minR_BW.txt")
minPBW  = np.genfromtxt("minP_BW.txt")
instxBW = np.genfromtxt("xinitBW.txt").reshape(-1,dof)
instoBW = np.concatenate([instxBW.reshape(-1,dof),minPBW.reshape(1,dof)],axis=0)
instpBW = np.concatenate([minRBW.reshape(1,dof),instoBW],axis=0)

T = 1.5e+3
tau = T / N

# Add pip1 in last voxel to path via constraint
pip1pFW = ρ - np.sum(instpFW[:,::species],axis=1) - np.sum(instpFW[:,4::species],axis=1)
instpFW_full = np.concatenate([instpFW,pip1pFW.reshape(N+1,1)],axis=1)
instpFW_full = instpFW_full.reshape(N+1,dofx,dofy,species)

pip1pBW = ρ - np.sum(instpBW[:,::species],axis=1) - np.sum(instpBW[:,4::species],axis=1)
instpBW_full = np.concatenate([instpBW,pip1pBW.reshape(N+1,1)],axis=1)
instpBW_full = instpBW_full.reshape(N+1,dofx,dofy,species)

#Extract PIP2 component
pip2pathFW = instpFW_full[:,:,:,0]
pip2pathBW = instpBW_full[:,:,:,0]

### Plotting ###
fig = plt.figure(figsize=(18, 13)) 

gs = GridSpec(nrows=4, ncols=6, figure=fig, width_ratios  = [1, 1, 1, 1, 1, 0.05], height_ratios = [1, 0.1,0.6, 0.6],
    wspace=0.10, hspace=0.20)

top = gs[0, 0:6].subgridspec(nrows=1, ncols=2, width_ratios=[1, 1.0], wspace=0.4)
ax2 = fig.add_subplot(top[0, 0])
ax1 = fig.add_subplot(top[0, 1])

####################### Stability diagram ###########################
ax1.set_xlabel('$\kappa$', size=24,labelpad=8)
ax1.set_ylabel('$\Omega^{-1}$', size=24, labelpad=12)
ax1.xaxis.set_major_locator(MultipleLocator(0.1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))

DM, ΩM = np.meshgrid(x_smooth,1/Ωarray)

Z_data = (fw_r > bw_r).astype(int) 
colormap = plt.get_cmap("coolwarm") 
cm_blue = mcolors.to_hex(colormap(0.0))  
cm_red = mcolors.to_hex(colormap(1.0)) 
colors = [cm_blue, cm_red]
cmap = matplotlib.colors.ListedColormap(colors)

contour_plot = ax1.contourf(DM, ΩM, Z_data.T, levels=[0, 0.5, 1], cmap=cmap) 
cbar = fig.colorbar(contour_plot, ax=ax1, ticks=[0.25, 0.75], format=plt.FuncFormatter(lambda x, pos: 'PIP$_2$' if x > 0.5 else 'PIP$_1$'))

ax1.set_yscale('log')
ax1.set_xlim(0.12,0.5)

####################### Action plot ###########################
# Identify crossing point
fw_p        = fw_all[:,1]
bw_p        = bw_all[:,1]
p1_smooth   = -a1_smooth
p2_smooth   = -a2_smooth 
cross_index = np.argmax(p1_smooth < p2_smooth)

# Determine the minimum curve
y_min = np.minimum(p1_smooth, p2_smooth)
ax2.plot(fw_all[:,0]  , -fw_p, '-o', color=cm_blue ,label='$\mathrm{PIP}_1 \\rightarrow \mathrm{PIP}_2$', lw=4.0, markersize=10, zorder=3)	
ax2.plot(bw_all[:,0]  , -bw_p, '-o', color=cm_red  ,label='$\mathrm{PIP}_2 \\rightarrow \mathrm{PIP}_1$', lw=4.0, markersize=10, zorder=3)	

ax2.set_xlabel('$\kappa$', size=24,labelpad=8)
ax2.set_ylabel('$-S$', size=24, labelpad=12)

ax2.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.2,1.0),fancybox=True,shadow=True)

ax2.set_xlim(np.min(fw_a[:,0]),0.5)
ax2.set_ylim(-1.67,-0.81)

y_bottom = ax2.get_ylim()[0]
ax2.fill_between(x=x_smooth[:cross_index], y1=y_min[:cross_index], y2=y_bottom, color=cm_red , alpha=0.5)
ax2.fill_between(x=x_smooth[cross_index:], y1=y_min[cross_index:], y2=y_bottom, color=cm_blue, alpha=0.5)


ax2.text(0.14 , -1.62, "PIP$_2$ stable", fontsize=24, color='k')
ax2.text(0.325, -1.62, "PIP$_1$ stable", fontsize=24, color='k')


ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(0.1*2))
ax2.yaxis.set_minor_locator(MultipleLocator(0.05*2))

########################### PATHS ################################
vmin, vmax = 0.05, 0.87

axesX = [fig.add_subplot(gs[2, i]) for i in range(5)]
caxX  = fig.add_subplot(gs[2, 5])

axesY = [fig.add_subplot(gs[3, i]) for i in range(5)]
caxY  = fig.add_subplot(gs[3, 5])

levels = np.linspace(vmin, vmax, 11)
x = np.arange(dofx + 1)
y = np.arange(dofy + 1)

# Plot each panel
idxs = [0, N//8, N//2, 7*N//8, -1]

 
for inda, idx in enumerate(idxs):
    # original grid
    X, Y = np.meshgrid(x[:-1], y[:-1], indexing='ij')
    Z = pip2pathFW[idx]
    # finer grid
    xf = np.linspace(x[:-1].min(), x[:-1].max(), 16*len(x))
    yf = np.linspace(y[:-1].min(), y[:-1].max(), 16*len(y))
    Xf, Yf = np.meshgrid(xf, yf, indexing='ij')
    # interpolate
    interp = RegularGridInterpolator((x[:-1], y[:-1]), Z)
    Zf = interp((Xf, Yf))
    # plot fw
    contX = axesX[inda].contourf(xf, yf, Zf, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax, extend='both')
    #contX = axesX[inda].contourf(x[:-1], y[:-1], instpFW[idx], levels=levels, cmap=colormap,vmin=vmin, vmax=vmax,extend='both')
    # plot bw
    Z = pip2pathBW[idx]
    interp = RegularGridInterpolator((x[:-1], y[:-1]), Z)
    Zf = interp((Xf, Yf))
    contY = axesY[inda].contourf(xf, yf, Zf, levels=levels, cmap=colormap, vmin=vmin, vmax=vmax, extend='both')

    axesX[inda].set_aspect("equal",adjustable="box")
    axesY[inda].set_aspect("equal", adjustable="box")
    axesY[inda].set_xlabel("$x_0$", size=24)

axesX[0].set_ylabel("$x_1$", size=24, labelpad=12)
axesY[0].set_ylabel("$x_1$", size=24, labelpad=12)

cbar = fig.colorbar(contX, cax=caxX, ticks=[0.05, 0.46, 0.87])
cbar = fig.colorbar(contY, cax=caxY, ticks=[0.05, 0.46, 0.87])
fig.canvas.draw()  # needed to compute positions

pos_ax = axesX[0].get_position()      # bounding box of a subplot
pos_cb = caxX.get_position()
caxX.set_position([pos_cb.x0, pos_ax.y0, pos_cb.width, pos_ax.height])

pos_ax = axesY[0].get_position()      
pos_cb = caxY.get_position()

caxY.set_position([pos_cb.x0, pos_ax.y0, pos_cb.width, pos_ax.height])

axesY[0].xaxis.set_major_locator(MultipleLocator(4.0))
axesY[0].xaxis.set_minor_locator(MultipleLocator(2.0))
axesY[0].yaxis.set_major_locator(MultipleLocator(4.0))
axesY[0].yaxis.set_minor_locator(MultipleLocator(2.0))

axesY[1].xaxis.set_major_locator(MultipleLocator(4.0))
axesY[1].xaxis.set_minor_locator(MultipleLocator(2.0))
axesY[1].yaxis.set_major_locator(MultipleLocator(4.0))
axesY[1].yaxis.set_minor_locator(MultipleLocator(2.0))

axesY[2].xaxis.set_major_locator(MultipleLocator(4.0))
axesY[2].xaxis.set_minor_locator(MultipleLocator(2.0))
axesY[2].yaxis.set_major_locator(MultipleLocator(4.0))
axesY[2].yaxis.set_minor_locator(MultipleLocator(2.0))

axesY[3].xaxis.set_major_locator(MultipleLocator(4.0))
axesY[3].xaxis.set_minor_locator(MultipleLocator(2.0))
axesY[3].yaxis.set_major_locator(MultipleLocator(4.0))
axesY[3].yaxis.set_minor_locator(MultipleLocator(2.0))

axesY[4].xaxis.set_major_locator(MultipleLocator(4.0))
axesY[4].xaxis.set_minor_locator(MultipleLocator(2.0))
axesY[4].yaxis.set_major_locator(MultipleLocator(4.0))
axesY[4].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[0].xaxis.set_major_locator(MultipleLocator(4.0))
axesX[0].xaxis.set_minor_locator(MultipleLocator(2.0))
axesX[0].yaxis.set_major_locator(MultipleLocator(4.0))
axesX[0].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[1].xaxis.set_major_locator(MultipleLocator(4.0))
axesX[1].xaxis.set_minor_locator(MultipleLocator(2.0))
axesX[1].yaxis.set_major_locator(MultipleLocator(4.0))
axesX[1].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[2].xaxis.set_major_locator(MultipleLocator(4.0))
axesX[2].xaxis.set_minor_locator(MultipleLocator(2.0))
axesX[2].yaxis.set_major_locator(MultipleLocator(4.0))
axesX[2].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[3].xaxis.set_major_locator(MultipleLocator(4.0))
axesX[3].xaxis.set_minor_locator(MultipleLocator(2.0))
axesX[3].yaxis.set_major_locator(MultipleLocator(4.0))
axesX[3].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[4].xaxis.set_major_locator(MultipleLocator(4.0))
axesX[4].xaxis.set_minor_locator(MultipleLocator(2.0))
axesX[4].yaxis.set_major_locator(MultipleLocator(4.0))
axesX[4].yaxis.set_minor_locator(MultipleLocator(2.0))

axesX[0].set_xticklabels([])
axesX[1].set_xticklabels([])
axesX[2].set_xticklabels([])
axesX[3].set_xticklabels([])
axesX[4].set_xticklabels([])

axesX[1].set_yticklabels([])
axesX[2].set_yticklabels([])
axesX[3].set_yticklabels([])
axesX[4].set_yticklabels([])

axesY[1].set_yticklabels([])
axesY[2].set_yticklabels([])
axesY[3].set_yticklabels([])
axesY[4].set_yticklabels([])

########################################

ax1.text(1.15,0.9,'(b)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.90,0.9,'(a)',transform=ax2.transAxes,fontsize=24)
axesX[0].text(-0.60,0.9,'(c)',transform=axesX[0].transAxes,fontsize=24)
axesY[0].text(-0.60,0.9,'(d)',transform=axesY[0].transAxes,fontsize=24)


plt.savefig("2DEnzyme_Combine.pdf", bbox_inches="tight", format='pdf',dpi=600)
plt.show()


