#!/usr/bin/env python
# coding: utf-8


""" SI Fig. 7 """
__author__ = 'Eric Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'


### KMC FW rate ###
kmcFW = np.array([
6  ,  0.00500 ,   0.00488 ,   0.00482 ,   0.00559 ,   0.00493 ,
22 ,  0.00164 ,   0.00165 ,   0.00177 ,   0.00169 ,   0.00182 ,
50 ,  0.000709,   0.000729,   0.000747,   0.000655,   0.000741,
89 ,  0.000255,   0.000242,   0.000272,   0.000234,   0.000250,
139,  7.37e-05,   7.67e-05,   8.10e-05,   7.48e-05,   7.06e-05,
200,  1.87e-05,   1.78e-05,   1.74e-05,   1.84e-05,   1.73e-05,
235,  8.24e-06,   7.99e-06,   7.17e-06,   7.22e-06,   7.59e-06,
272,  3.27e-06,   3.26e-06,   3.38e-06,   3.10e-06,   3.62e-06
])


### CLE FW rate ###
kcleFW = np.array([
6  ,   0.0113  ,   0.0137  ,   0.0131  ,   0.0126  ,   0.0119  ,
13 ,   0.00395 ,   0.00459 ,   0.00477 ,   0.00461 ,   0.00418 ,
22 ,   0.00242 ,   0.00214 ,   0.00242 ,   0.00227 ,   0.00207 ,
35 ,   0.00122 ,   0.00120 ,   0.00137 ,   0.00126 ,   0.00113 ,
50 ,   0.000817,   0.000752,   0.000792,   0.000781,   0.000818,
68 ,   0.000439,   0.000431,   0.000512,   0.000447,   0.000412,
89 ,   0.000341,   0.000255,   0.000262,   0.000234,   0.000317,
113,   0.000179,   0.000147,   0.000126,   0.000160,   0.000139,
139,   9.15e-05,   8.21e-05,   7.76e-05,   9.55e-05,   7.80e-05,
168,   4.19e-05,   4.17e-05,   3.80e-05,   4.24e-05,   3.76e-05,
200,   1.63e-05,   1.96e-05,   1.82e-05,   2.10e-05,   1.74e-05,
235,   9.13e-06,   8.19e-06,   7.30e-06,   7.02e-06,   7.79e-06,
272,   3.19e-06,   3.00e-06,   3.43e-06,   3.31e-06,   3.68e-06,
313,   1.48e-06,   1.38e-06,   1.32e-06,   1.36e-06,   1.22e-06
])


### KMC BW rate ###
kmcBW = np.array([
6  ,  0.00390 ,   0.00392 ,   0.00408 ,   0.00413 ,   0.00364 ,
22 ,  0.00283 ,   0.00284 ,   0.00287 ,   0.00339 ,   0.00311 ,
50 ,  0.00112 ,   0.00106 ,   0.00103 ,   0.00106 ,   0.00108 ,
89 ,  0.000352,   0.000376,   0.000342,   0.000362,   0.000319,
139,  0.000120,   0.000137,   9.73e-05,   0.000116,   0.000105,
200,  3.08e-05,   3.69e-05,   3.21e-05,   3.59e-05,   3.12e-05
])


### CLE BW rate ###
kcleBW = np.array([
6  ,  0.0109  ,   0.0101  ,   0.0103  ,   0.0102  ,   0.0108  ,
13 ,  0.00530 ,   0.00459 ,   0.00538 ,   0.00488 ,   0.00475 ,
22 ,  0.00314 ,   0.00267 ,   0.00285 ,   0.00269 ,   0.00287 ,
35 ,  0.00169 ,   0.00143 ,   0.00176 ,   0.00162 ,   0.00174 ,
50 ,  0.000798,   0.000896,   0.00116 ,   0.000978,   0.000956,
68 ,  0.000668,   0.000659,   0.000660,   0.000615,   0.000653,
89 ,  0.000381,   0.000376,   0.000427,   0.000501,   0.000387,
113,  0.000238,   0.000193,   0.000227,   0.000219,   0.000243,
139,  0.000105,   0.000127,   0.000128,   0.000123,   0.000128,
168,  5.61e-05,   6.13e-05,   6.73e-05,   6.36e-05,   6.53e-05,
200,  2.91e-05,   2.85e-05,   2.90e-05,   3.85e-05,   2.84e-05,
235,  1.37e-05,   1.45e-05,   1.70e-05,   1.46e-05,   1.64e-05,
272,  5.54e-06,   5.83e-06,   6.47e-06,   7.11e-06,   5.57e-06,
313,  2.54e-06,   2.61e-06,   2.45e-06,   2.86e-06,   2.51e-06
])

kmcFW    = kmcFW.reshape(-1,6)
kcleFW   = kcleFW.reshape(-1,6)
kmcBW    = kmcBW.reshape(-1,6)
kcleBW   = kcleBW.reshape(-1,6)

Ω_kmcFW  = kmcFW[:,0]
kmc_rFW  = kmcFW[:,1:]
kmc_mFW  = np.mean(kmc_rFW,axis=1)
kmc_sFW  = np.std(kmc_rFW,axis=1)

Ω_cleFW  = kcleFW[:,0]
kcle_rFW = kcleFW[:,1:]
kcle_mFW = np.mean(kcle_rFW,axis=1)
kcle_sFW = np.std(kcle_rFW ,axis=1)

Ω_kmcBW  = kmcBW[:,0]
kmc_rBW  = kmcBW[:,1:]
kmc_mBW  = np.mean(kmc_rBW,axis=1)
kmc_sBW  = np.std(kmc_rBW,axis=1)

Ω_cleBW  = kcleFW[:,0]
kcle_rBW = kcleBW[:,1:]
kcle_mBW = np.mean(kcle_rBW,axis=1)
kcle_sBW = np.std(kcle_rBW ,axis=1)


#Convert instanton rates between different noise strengths
def convert(ko,S,Oo,On):
    kn = ko * np.exp(S*(Oo - On))
    return kn

# Instanton rates
### Forward
Ω_neqiFW = Ω_neqiBW = np.linspace(6,322,1000)
#
kFW     = 0.002075593430779385
SinstFW = 0.023941321278728885
OmFW    = 5.56
karrFW  = convert(kFW,SinstFW,OmFW,Ω_neqiFW)

#### Backward
kBW     = 0.002340124255489413
SinstBW = 0.022232252551798882
OmBW    = 5.56
karrBW  = convert(kBW,SinstBW,OmBW,Ω_neqiBW)

fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17,6))

ax1.plot(Ω_kmcFW , kmc_mFW , "--o", lw=3.0, color='k' , zorder=6, label='KMC')
ax1.plot(Ω_cleFW , kcle_mFW, "--o", lw=3.0, color='C3', zorder=5, label='CLE')
ax1.plot(Ω_neqiFW, karrFW  , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')


ax1.errorbar(Ω_kmcFW, kmc_mFW, yerr=2*kmc_sFW, fmt='o', color='k', markersize=6, elinewidth=2.5, zorder=5)
ax1.errorbar(Ω_cleFW, kcle_mFW, yerr=2*kcle_sFW, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

ax1.xaxis.set_major_locator(MultipleLocator(100))
ax1.xaxis.set_minor_locator(MultipleLocator(50))

ax1.set_yscale('log')

ax1.set_xlabel("$\Omega$",size=24,labelpad=8)
ax1.set_ylabel("$k$",size=24,labelpad=12)

ax1.legend()

#####################################################################3
ax2.plot(Ω_kmcBW , kmc_mBW , "--o", lw=3.0, color='k' , zorder=6, label='KMC')
ax2.plot(Ω_cleBW , kcle_mBW, "--o", lw=3.0, color='C3', zorder=5, label='CLE')
ax2.plot(Ω_neqiBW, karrBW  , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')


ax2.errorbar(Ω_kmcBW, kmc_mBW, yerr=2*kmc_sBW, fmt='o', color='k', markersize=6, elinewidth=2.5, capsize=3, zorder=5)
ax2.errorbar(Ω_cleBW, kcle_mBW, yerr=2*kcle_sBW, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_minor_locator(MultipleLocator(50))

ax2.set_yscale('log')

ax2.set_xlabel("$\Omega$",size=24,labelpad=8)
ax2.set_ylabel("$k$",size=24,labelpad=12)

plt.tight_layout()
fig.subplots_adjust(wspace=0.4)

plt.savefig("Enzyme_RateComparison.pdf", format='pdf',dpi=600)

plt.show()


