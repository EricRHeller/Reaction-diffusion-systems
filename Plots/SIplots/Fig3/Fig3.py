#!/usr/bin/env python
# coding: utf-8

""" SI Fig. 3 """
__author__ = 'Eric Heller'
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
matplotlib.rcParams['legend.fontsize'] = 24
matplotlib.rcParams['legend.loc'] = "lower left"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'


#### Actions ###
# Import actions
fw_all  = np.genfromtxt("action_fw_full.txt")
bw_all  = np.genfromtxt("action_bw_full.txt")

fw_all[:,1] *= -1
bw_all[:,1] *= -1

# Spline interpolate actions
aspline1 = make_interp_spline(fw_all[:,0], fw_all[:,1], k=3)
aspline2 = make_interp_spline(bw_all[:,0], bw_all[:,1], k=3)
x_smootha = np.linspace(np.min(fw_all[:,0]), np.max(fw_all[:,0]), 1001)  # Smooth range for spline fitting\n",
# Evaluate spline curves
a1_smooth = aspline1(x_smootha)
a2_smooth = aspline2(x_smootha)


### Rates ###
#Import
fw  = np.genfromtxt("fw.txt")
bw  = np.genfromtxt("bw.txt")

### Plotting ###
fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,6))

colormap = plt.get_cmap('coolwarm')
cm_blue = mcolors.to_hex(colormap(0.0))
cm_red  = mcolors.to_hex(colormap(1.0))

# Action 
ax1.set_xlabel('$\kappa$', size=24,labelpad=8)

ax1.xaxis.set_major_locator(MultipleLocator(4.0))
ax1.xaxis.set_minor_locator(MultipleLocator(2.0))
ax1.yaxis.set_major_locator(MultipleLocator(0.05))
ax1.yaxis.set_minor_locator(MultipleLocator(0.025))

ax1.plot(fw_all[:,0]  , fw_all[:,1], '-o', color=cm_blue ,label=r'$\rho_{-} \to \rho_{+}$', lw=3.0, zorder=3, markersize=8)	
ax1.plot(bw_all[:,0]  , bw_all[:,1], '-o', color=cm_red  ,label=r'$\rho_{+} \to \rho_{-}$', lw=3.0, zorder=3, markersize=8)	

ax1.set_ylabel('$-S$', size=24, labelpad=12)
ax1.set_xlim(2.0,17.0)
ax1.set_ylim(-0.3,-0.1)

# Identify crossing point
cross_index = np.argmax(a1_smooth < a2_smooth)
# Determine the minimum curve\n",
y_min = np.minimum(a1_smooth, a2_smooth)
y_bottom = ax1.get_ylim()[0]
ax1.fill_between(x=x_smootha[:cross_index], y1=y_min[:cross_index], y2=y_bottom, color=cm_red , alpha=0.5)
ax1.fill_between(x=x_smootha[cross_index:], y1=y_min[cross_index:], y2=y_bottom, color=cm_blue, alpha=0.5)

leg = ax1.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.3, 0.95), fancybox=True, shadow=True)
ax1.text(2.4 , -0.27, r'\begin{center} $\rho_{+}$\\ \vspace{0.1cm} stable \end{center}',fontsize=24,color='k')
ax1.text(8.0,  -0.27, r'\begin{center} $\rho_{-}$\\ \vspace{0.1cm} stable \end{center}',fontsize=24,color='k')

################ Rates ##################
ax2.plot(fw[:,0]  , fw[:,1], '-o', color=cm_blue ,label='$\rho_{-}\\rightarrow \rho_{+}$', lw=3.0, zorder=3)	
ax2.plot(bw[:,0]  , bw[:,1], '-o', color=cm_red  ,label='$\rho_{+}\\rightarrow \rho_{-}$', lw=3.0, zorder=3)	

ax2.set_xlabel('$\kappa$', size=24,labelpad=8)
ax2.set_ylabel('$k$', size=24, labelpad=12)
ax2.set_yscale('log')
ax2.xaxis.set_major_locator(MultipleLocator(4.0))
ax2.xaxis.set_minor_locator(MultipleLocator(2.0))

# Fitting splines
x_smooth = np.linspace(np.min(fw[:,0]), np.max(fw[:,0]), 500)  # Smooth range for spline fitting
spline1 = make_interp_spline(fw[:,0], np.log(fw[:,1]), k=1)
spline2 = make_interp_spline(bw[:,0], np.log(bw[:,1]), k=1)

# Evaluate spline curves
y1_smooth = np.exp(spline1(x_smooth))
y2_smooth = np.exp(spline2(x_smooth))

# Determine the minimum curve
y_min = np.minimum(y1_smooth, y2_smooth)
ax2.fill_between(x=x_smooth, y1=y_min,y2=np.min(y_min), color=cm_blue, alpha=0.5)
#ax2.fill_between(x=x_smooth[cross_index:], y1=y_min[cross_index:],y2=np.min(y_min), color='b', alpha=0.5)

ax2.set_xlim(1.0,16.0)
ax2.set_ylim(np.min(y_min),1e+4)

#ax2.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.2,1.0),fancybox=True,shadow=True)
#ax2.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.6,1.0),fancybox=True,shadow=True)

ax2.text(2,1e-11,r"$\rho_{-}$ stable",fontsize=24,color='k')

##################################################

plt.tight_layout()
fig.subplots_adjust(wspace=0.4)

ax1.text(0.91,0.9,'(a)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.91,0.9,'(b)',transform=ax2.transAxes,fontsize=24)

plt.savefig("1DSchloegl_ratesaction.pdf", format='pdf', dpi=600)
plt.show()




