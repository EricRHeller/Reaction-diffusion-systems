#!/usr/bin/env python
# coding: utf-8

"""SI Fig. 2"""
__author__ = 'Eric Heller'
import numpy as np
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
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'


### Kappa=1 ###
# KMC rate
kmc1 = np.array([
50 ,  0.00329,  0.00331,  0.00360,  0.00373,  0.00325,
80 ,  0.000538, 0.000645, 0.000666, 0.000664, 0.000524,
100,  0.000164, 0.000154, 0.000140, 0.000143, 0.000156
])


# CLE rate
kcle1 = np.array([
5e+1   , 0.00390,  0.00364,  0.00373,  0.00327,  0.00385,
8e+1   , 0.000595, 0.000603, 0.000635, 0.000730, 0.000663,
1e+2   , 0.000167, 0.000185, 0.000166, 0.000161, 0.000183,
1.5e+2 , 3.92e-06, 4.36e-06, 4.12e-06, 3.46e-06, 3.72e-06,
2e+2   , 1.00e-07, 1.08e-07, 1.05e-07, 1.06e-07, 1.05e-07,
2.1e+2 , 4.17e-08, 3.90e-08, 4.10e-08, 3.93e-08, 4.13e-08,
2.15e+2, 3.12e-08, 3.17e-08, 2.93e-08, 3.12e-08, 3.12e-08,
2.2e+2 , 2.13e-08, 1.95e-08, 2.02e-08, 1.80e-08, 2.26e-08,
2.3e+2 , 8.80e-09, 8.90e-09, 8.99e-09, 8.75e-09, 8.29e-09
])


### kappa=8 ###
# KMC rate
kmc8 = np.array([
10, 0.0256  , 0.0246  , 0.0256  , 0.0244  , 0.0255  ,
30, 0.00287 , 0.00257 , 0.00279 , 0.00297 , 0.00271 ,
50, 0.000327, 0.000352, 0.000258, 0.000298, 0.000300
])


# CLE rate
kcle8 = np.array([
10, 0.0277  , 0.0293  , 0.0337  , 0.0326  , 0.0301  ,
30, 0.00291 , 0.00296 , 0.00302 , 0.00300 , 0.00317 ,
50, 0.000335, 0.000309, 0.000306, 0.000364, 0.000314,
60, 0.000105, 0.000112, 9.62e-05, 9.42e-05, 0.000108,
80, 1.03e-05, 1.14e-05, 1.05e-05, 1.03e-05, 1.00e-05
])


kmc1  = kmc1.reshape(-1,6)
kcle1 = kcle1.reshape(-1,6)

kmc_r1 = kmc1[:,1:]
kmc_m1 = np.mean(kmc_r1,axis=1)
kmc_s1 = np.std(kmc_r1,axis=1)

kcle_r1 = kcle1[:,1:]
kcle_m1 = np.mean(kcle_r1,axis=1)
kcle_s1 = np.std(kcle_r1 ,axis=1)


kmc8   = kmc8.reshape(-1,6)
kcle8  = kcle8.reshape(-1,6)

kmc_r8 = kmc8[:,1:]
kmc_m8 = np.mean(kmc_r8,axis=1)
kmc_s8 = np.std(kmc_r8,axis=1)

kcle_r8 = kcle8[:,1:]
kcle_m8 = np.mean(kcle_r8,axis=1)
kcle_s8 = np.std(kcle_r8 ,axis=1)


### Instanton rates ###
def convert(ko,S,Oo,On,interface=True):
    if interface==True:
        kn = ko * np.exp(S*(Oo - On)) * np.sqrt(On/Oo) 
    else:
        kn = ko * np.exp(S*(Oo - On))
    return kn

# kappa=1
k50_1     = 0.012245070048999015
Sinst1    = 0.08167780393060768
om1       = 50
oarray1   = np.linspace(0.1*om1,6*om1,100)
karr1     = convert(k50_1,Sinst1,om1,oarray1,interface=True)

# kappa=8
k50_8     = 0.00041282224746025116
Sinst8    = 0.1163806375776572
om8       = 50
oarray8   = np.linspace(0.1*om8,4*om8,100)
karr8     = convert(k50_8,Sinst8,om8,oarray8,interface=False)


fig,ax = plt.subplots(figsize=(8.5,6))

### kappa=1 ###
plt.scatter(kmc1[:,0] , kmc_m1 , s=50        , color='k' , zorder=6, label='KMC')
plt.plot(   kcle1[:,0], kcle_m1, "--", lw=3.0, color='C3', zorder=5, label='CLE')
plt.plot(   oarray1   , karr1  , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')

plt.errorbar(kmc1[:, 0], kmc_m1, yerr=2*kmc_s1, fmt='o', color='k',  markersize=6, elinewidth=2.5, capsize=3, zorder=5)
plt.errorbar(kcle1[:, 0], kcle_m1, yerr=2*kcle_s1, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

### kappa=8 ###
plt.scatter(kmc8[:,0] , kmc_m8 , s=50        , facecolors='none', edgecolors='k', zorder=6, alpha=0.6)
plt.plot(   oarray8   , karr8  , "-" , lw=3.0, color='C2', zorder=4, alpha=0.6)
plt.plot(   kcle8[:,0], kcle_m8, "--", lw=3.0, color='C3', zorder=5, alpha=0.6)

plt.errorbar(kmc8[:, 0], kmc_m8, yerr=2*kmc_s8, fmt='o', color='k', markersize=6, elinewidth=2.5, capsize=3, zorder=5, alpha=0.6)
plt.errorbar(kcle8[:, 0], kcle_m8, yerr=2*kcle_s8, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5, alpha=0.6)


plt.yscale('log')

plt.xlabel("$\Omega$",size=24,labelpad=8)
plt.ylabel("$k$",size=24,labelpad=12)

plt.xlim(5,240)
plt.ylim(5e-9,1e-1)
#
ax.xaxis.set_major_locator(MultipleLocator(40))
ax.xaxis.set_minor_locator(MultipleLocator(20))

plt.legend()

plt.tight_layout()

plt.savefig("1DSchloegl_benchmark.pdf", format='pdf', dpi=600)

plt.show()


