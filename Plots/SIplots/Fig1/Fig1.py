#!/usr/bin/env python
# coding: utf-8

""" SI. Fig 1 """
__author__ = 'Eric Heller'
import numpy as np
from scipy.integrate import simps
from scipy import optimize
import matplotlib
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
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['legend.loc'] = "lower left"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'


# KMC rates
kmc = np.array([
50  , 0.0554  , 0.0503  , 0.0537  , 0.0509  , 0.0495  ,
100 , 0.0271  , 0.0239  , 0.0258  , 0.0252  , 0.0240  ,
200 , 0.00916 , 0.00934 , 0.00984 , 0.0105  , 0.0102  ,
400 , 0.00194 , 0.00214 , 0.00183 , 0.00186 , 0.00200 ,
600 , 0.000493, 0.000527, 0.000454, 0.000442, 0.000402,
800 , 0.000116, 0.000107, 0.000104, 9.24e-05, 9.79e-05,
1000, 2.67e-05, 2.15e-05, 2.50e-05, 2.31e-05, 2.60e-05
])


# CLE rates
kcle = np.array([
50,   0.0584  , 0.0599  , 0.0613  , 0.0621  , 0.0645  ,
100,  0.0272  , 0.0295  , 0.0291  , 0.0269  , 0.0279  ,
200,  0.0107  , 0.0114  , 0.0104  , 0.0110  , 0.0109  ,
400,  0.00224 , 0.00203 , 0.00249 , 0.00202 , 0.00227 ,
600,  0.000490, 0.000487, 0.000513, 0.000527, 0.000539,
800,  0.000122, 0.000107, 0.000140, 0.000127, 0.000105,
1000, 2.80e-05, 2.55e-05, 2.74e-05, 2.64e-05, 2.80e-05
])


kmc  = kmc.reshape(-1,6)
kcle = kcle.reshape(-1,6)

kmc_r = kmc[:,1:]
kmc_m = np.mean(kmc_r,axis=1)
kmc_s = np.std(kmc_r,axis=1)

kcle_r = kcle[:,1:]
kcle_m = np.mean(kcle_r,axis=1)
kcle_s = np.std(kcle_r,axis=1)

#Convert instanton rates between different noise strengths
def convert(ko,S,Oo,On):
    kn = ko*np.exp(S*(Oo - On))
    return kn

# Instanton results
k100  = 0.0199769434219721
Sinst = 0.007287858860199874

om = 100
oarray = np.linspace(0.1*om,10*om,100)
karr   = convert(k100,Sinst,om,oarray)

#Effective potential plot
dof = 1
#
kp1 = 0.8 
kp2 = 3.1
km1 = 2.9
km2 = 1

# Noise matrix
def Dinvf(x):
    matD = kp2*x**2 + km2*x**3 + km1*x + kp1
    Dinv = 1.0/matD 
    return Dinv

# Noise-scaled drift
def Ftilde(x):
    f    = force(x)
    Dinv = Dinvf(x)
    ft   = Dinv * f
    return ft

### Potential generating conservative force ###
def potential(x):
    V = -kp2*x[0]**3/3 + km2*x[0]**4/4 + km1*x[0]**2/2 - kp1*x[0] 
    return V

######
def force(x):
    f = kp2*x**2 - km2*x**3 - km1*x + kp1 
    return f

def dforcedx(x):
    h = 2*kp2*x - 3*km2*x**2 - km1 
    return h

### Locate minima and TS ###
print("Locate reactant and product minima, as well as the TS")
# Reactant
guess = 0.5
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
minR = sol.x
jacR = dforcedx(minR)
print("minR",minR)

# Product
guess = 1.6
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
minP = sol.x
jacP = dforcedx(minP)
print("minP",minP)

# TS
guess = 1.0
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
TS = sol.x
jacTS = dforcedx(TS)
print("TS",TS)

# Integrated drift
xplot = np.linspace(0.01,2,1001)
Vplot = np.zeros_like(xplot)
for indx,xi in enumerate(xplot):
    Vplot[indx] = potential(np.array([xi]))
Vplot -= potential(minR)

# Effective potential or quasipotential
Veff = np.zeros_like(xplot)
for indx,xi in enumerate(xplot[1:]):
    intgrid = np.linspace(minR[0],xi,1000)
    ftild = 2*Ftilde(intgrid)
    Veff[indx] = -simps(ftild,intgrid)


### Plot 
fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,6))

ax2.scatter(kmc[:,0],kmc_m,s=50,color='k',zorder=6,label='KMC')
ax2.plot(oarray   , karr  , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')
ax2.plot(kcle[:,0], kcle_m, "--", lw=3.0, color='C3', zorder=5, label='CLE')

ax2.errorbar(kmc[:, 0], kmc_m, yerr=2*kmc_s, fmt='o', color='k', markersize=6, elinewidth=2.5, capsize=3,  zorder=5)
ax2.errorbar(kcle[:, 0], kcle_m, yerr=2*kcle_s, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

ax2.set_yscale('log')

ax2.set_xlabel("$\Omega$",size=24,labelpad=8)
ax2.set_ylabel("$k$",size=24,labelpad=12)

ax2.legend()


############ Eff potential ####################
ax1.set_xlabel(r"$\rho$",fontsize=24,labelpad=8)
ax1.set_ylabel("$V_\mathrm{eff}$",fontsize=24,labelpad=12,color='C1')

ax1.plot(xplot,Veff,'-',lw=3.0,color='C1', label='Int Ftilde')

ax1.set_xlim(0.25,1.9)
ax1.set_ylim(-2*0.0002,2*0.0045)
#
#
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.xaxis.set_minor_locator(MultipleLocator(0.25))
ax1.yaxis.set_major_locator(MultipleLocator(0.002))
ax1.yaxis.set_minor_locator(MultipleLocator(0.001))


### Second Plot ###
# Create second y-axis
axins = ax1.twinx()
axins.set_ylabel("$V$",fontsize=24,labelpad=-6,color='C0')

axins.plot(xplot,Vplot,lw=3.0,color='C0',label='Potential')
axins.set_ylim(-0.015,0.03)
axins.yaxis.set_major_locator(MultipleLocator(0.01))
axins.yaxis.set_minor_locator(MultipleLocator(0.005))


### Colour axes ###
ax1.tick_params(axis='y', labelcolor='C1')
axins.tick_params(axis='y', labelcolor='C0')

##################################################

plt.tight_layout()
fig.subplots_adjust(wspace=0.5)

ax1.text(0.91,0.9,'(a)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.91,0.9,'(b)',transform=ax2.transAxes,fontsize=24)

plt.savefig("Schloegl_combine.pdf", format='pdf', dpi=600)
plt.show()


