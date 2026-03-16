#!/usr/bin/env python
# coding: utf-8

""" SI Fig. 8 """
__author__ = 'Eric Heller'
import numpy as np
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
def convert(ko,S,Oo,On,interface=True):
    if interface==True:
        kn = ko + 0.5 * np.log(On/Oo) + S*(Oo - On)
    else:
        kn = ko + S*(Oo - On)
    return kn


### Import data files
# Rates
fw           = np.genfromtxt("fw.txt")
bw           = np.genfromtxt("bw.txt")
# Actions matching rates
fw_a         = np.genfromtxt("action_fw.txt")
bw_a         = np.genfromtxt("action_bw.txt")
# All actions
fw_all       = np.genfromtxt("action_fw_full.txt")
bw_all       = np.genfromtxt("action_bw_full.txt")

actions_fw   = fw_a[:,1]
actions_bw   = bw_a[:,1]

# Spline interpolate actions
aspline1     = make_interp_spline(fw_all[:,0], fw_all[:,1], k=3)
aspline2     = make_interp_spline(bw_all[:,0], bw_all[:,1], k=3)

x_smooth     = np.linspace(np.min(fw[:,0]), np.max(fw[:,0]), 1001)  # Smooth range for spline fitting
# Evaluate spline curves
a1_smooth    = aspline1(x_smooth)
a2_smooth    = aspline2(x_smooth)

# Compute instanton rates on the kappa-Omega grid
fw_r = np.zeros((len(x_smooth),len(Ωarray)))
bw_r = np.zeros((len(x_smooth),len(Ωarray)))
fw_pre = np.log(fw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
bw_pre = np.log(bw[:,1].reshape(-1,1) * np.ones_like(Ωarray).reshape(1,-1))
for indo,Ωi in enumerate(Ωarray):
    for indr in range(len(actions_fw)):
        if indr < 4:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Ωi,interface=True)
        else:
            fw_pre[indr,indo] = convert(fw_pre[indr,indo],actions_fw[indr],Ω,Ωi,interface=False) 
    for indr in range(len(actions_bw)):
        if indr < 4:
            bw_pre[indr,indo] = convert(bw_pre[indr,indo],actions_bw[indr],Ω,Ωi,interface=True)
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
N           = 1024
# Physical parameters
species     = 5                   # Number of species
voxels      = 8                   # Number of voxels
dof         = species*voxels - 1  # Degrees of freedom (minus constraint)


# Kappa=12
minR12  = np.genfromtxt("minR_12.txt")
minP12  = np.genfromtxt("minP_12.txt")
instx12 = np.genfromtxt("xinit12.txt").reshape(-1,dof)
ionex12 = np.concatenate([instx12.reshape(-1,dof),minP12.reshape(1,dof)],axis=0)
instp12 = np.concatenate([minR12.reshape(1,dof),ionex12],axis=0)

# Kappa=60
minR60  = np.genfromtxt("minR_60.txt")
minP60  = np.genfromtxt("minP_60.txt")
instx60 = np.genfromtxt("xinit60.txt").reshape(-1,dof)
ionex60 = np.concatenate([instx60.reshape(-1,dof),minP60.reshape(1,dof)],axis=0)
instp60 = np.concatenate([minR60.reshape(1,dof),ionex60],axis=0)


T = 1e+3
tau = T / N


fig = plt.figure(figsize=(18, 13)) 

gs = GridSpec(nrows=3, ncols=4, figure=fig, width_ratios  = [1, 0.2, 1, 0.05], height_ratios = [1, 0.1, 1.0], wspace=0.10, 
              hspace=0.20)
top = gs[0, 0:4].subgridspec(nrows=1, ncols=2, width_ratios=[1, 1.0], wspace=0.4)

ax2 = fig.add_subplot(top[0, 0])
ax1 = fig.add_subplot(top[0, 1])

####################### Stability diagram ###########################
ax1.set_xlabel('$\kappa$', size=24,labelpad=12)
ax1.set_ylabel('$\Omega^{-1}$', size=24, labelpad=12)
ax1.xaxis.set_major_locator(MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))

DM, ΩM = np.meshgrid(x_smooth,1/Ωarray)

Z_data = (fw_r > bw_r).astype(int) 
colormap = plt.get_cmap("coolwarm") 
cm_blue = mcolors.to_hex(colormap(0.0))   
cm_red = mcolors.to_hex(colormap(1.0))
colors = [cm_blue, cm_red]
cmap = matplotlib.colors.ListedColormap(colors)

# Visualize the result
contour_plot = ax1.contourf(DM, ΩM, Z_data.T, levels=[0, 0.5, 1], cmap=cmap) 
cbar = fig.colorbar(contour_plot, ax=ax1, ticks=[0.25, 0.75], format=plt.FuncFormatter(lambda x, pos: 'PIP$_2$' if x > 0.5 else 'PIP$_1$'))

ax1.set_yscale('log')
ax1.set_xlim(0.1,0.9)

####################### Action plot ###########################
fw_a  = np.genfromtxt("action_fw_full.txt")
bw_a  = np.genfromtxt("action_bw_full.txt")

# Fitting splines
x_smootha = np.linspace(np.min(fw_a[:,0]), np.max(fw_a[:,0]), 10000) 
spline1 = make_interp_spline(fw_a[:,0], -fw_a[:,1], k=3)
spline2 = make_interp_spline(bw_a[:,0], -bw_a[:,1], k=3)

# Evaluate spline curves
y1_smooth = spline1(x_smootha)
y2_smooth = spline2(x_smootha)
# Identify crossing point
cross_index = np.argmax(y1_smooth < y2_smooth)

# Determine the minimum curve
y_min = np.minimum(y1_smooth, y2_smooth)

ax2.set_xlabel('$\kappa$', size=24,labelpad=12)
ax2.set_ylabel('$-S$', size=24, labelpad=12)

### Actions ###
ax2.plot(fw_a[:,0]  , -fw_a[:,1], '-o', color=cm_blue ,label='PIP$_1$ $\\rightarrow$ PIP$_2$', lw=3.0, zorder=3)
ax2.plot(bw_a[:,0]  , -bw_a[:,1], '-o', color=cm_red ,label='PIP$_2$ $\\rightarrow$ PIP$_1$', lw=3.0, zorder=3)

ax2.set_xlabel('$\kappa$', size=24,labelpad=12)
ax2.set_ylabel('$-S$'    , size=24, labelpad=12)

ax2.set_xlim(np.min(fw_a[:,0]),0.4)
ax2.set_ylim(-0.201, -0.126)

y_bottom = ax2.get_ylim()[0]
ax2.fill_between(x=x_smootha[:cross_index], y1=y_min[:cross_index], y2=y_bottom, color=cm_red , alpha=0.5)
ax2.fill_between(x=x_smootha[cross_index:], y1=y_min[cross_index:], y2=y_bottom, color=cm_blue, alpha=0.5)

ax2.text(0.13 , -0.196,"PIP$_2$ stable",fontsize=24,color='k')
ax2.text(0.27 , -0.196,"PIP$_1$ stable",fontsize=24,color='k')

ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(MultipleLocator(0.02))
ax2.yaxis.set_minor_locator(MultipleLocator(0.01))

ax2.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.27,1.0),fancybox=True,shadow=True)

########################### PATHS ################################
vmin, vmax = 0.05, 0.87

axesX = [fig.add_subplot(gs[2, i]) for i in range(3)]
caxX  = fig.add_subplot(gs[2, 3])

pip2path12 = instp12.reshape(N+1,dof)[:,::species]
pip2path12 = np.roll(pip2path12,0,axis=1)

pip2path60 = instp60.reshape(N+1,dof)[:,::species]
pip2path60 = np.roll(pip2path60,0,axis=1)

####
#Plot
tplot = np.linspace(0.0,T,N+1)
xplot = np.linspace(0.0,voxels-1,voxels)

TIME, X = np.meshgrid(tplot, xplot)

vmin, vmax = 0.05, 0.87
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
levels = np.linspace(vmin, vmax, 21)

cm12 = axesX[0].contourf(TIME, X, pip2path12.T, levels=levels, norm=norm, extend='both', cmap=colormap)
cm60 = axesX[2].contourf(TIME, X, pip2path60.T, levels=levels, norm=norm, extend='both', cmap=colormap)

axesX[1].axis('off')

axesX[0].set_xlabel("$t$", size=24, labelpad=12)
axesX[2].set_xlabel("$t$", size=24, labelpad=12)

axesX[0].set_ylabel("$x$", size=24, labelpad=12)

axesX[0].set_xlim(np.min(tplot), np.max(tplot))
axesX[0].set_ylim(np.min(xplot), np.max(xplot))

axesX[2].set_xlim(np.min(tplot), np.max(tplot))
axesX[2].set_ylim(np.min(xplot), np.max(xplot))

axesX[2].set_yticklabels([])
cbar = fig.colorbar(cm60, cax=caxX, ticks=[0.05, 0.46, 0.87])
cbar.ax.tick_params(labelsize=24)
cbar.set_label(r"$\rho_{\mathrm{PIP}_2}(x,t)$", fontsize=24, labelpad=12)

fig.canvas.draw()  

pos_ax = axesX[0].get_position()     
pos_cb = caxX.get_position()

caxX.set_position([pos_cb.x0, pos_ax.y0, pos_cb.width, pos_ax.height])

axesX[0].xaxis.set_major_locator(MultipleLocator(200))
axesX[0].xaxis.set_minor_locator(MultipleLocator(100))
axesX[0].yaxis.set_major_locator(MultipleLocator(2.0))
axesX[0].yaxis.set_minor_locator(MultipleLocator(1.0))

axesX[2].xaxis.set_major_locator(MultipleLocator(200))
axesX[2].xaxis.set_minor_locator(MultipleLocator(100))
axesX[2].yaxis.set_major_locator(MultipleLocator(2.0))
axesX[2].yaxis.set_minor_locator(MultipleLocator(1.0))
########################################

ax1.text(1.15,0.9,'(b)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.90,0.9,'(a)',transform=ax2.transAxes,fontsize=24)
axesX[0].text(-0.24,0.95,'(c)',transform=axesX[0].transAxes,fontsize=24)
axesX[2].text(-0.1,0.95,'(d)',transform=axesX[2].transAxes,fontsize=24)


plt.savefig("1DEnzyme_Combine.pdf", bbox_inches="tight", format='pdf',dpi=600)

plt.show()




