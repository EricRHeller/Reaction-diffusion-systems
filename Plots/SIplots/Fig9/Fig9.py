#!/usr/bin/env python
# coding: utf-8

""" SI Fig. 9 """
__author__ = 'Eric Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

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

############### Forward ##################

#KMC rates
FWkmc60 = np.array([
8 ,  0.00278 ,   0.00280 ,   0.00254 ,   0.00258 ,   0.00254 ,
13,  0.000949,   0.00110 ,   0.00103 ,   0.00127 ,   0.00103 ,
17,  0.000607,   0.000654,   0.000529,   0.000643,   0.000601,
24,  0.000163,   0.000177,   0.000200,   0.000178,   0.000202,
28,  9.05e-05,   0.000104,   0.000101,   9.96e-05,   9.92e-05,
34,  3.25e-05,   3.15e-05,   3.39e-05,   3.19e-05,   3.41e-05,
38,  1.55e-05,   1.34e-05,   1.43e-05,   1.60e-05,   1.38e-05,
44,  5.28e-06,   4.82e-06,   5.55e-06,   4.63e-06,   5.00e-06
])


#CLE rates
FWkcle60 = np.array([
5.56 ,  0.00299 ,   0.00292 ,   0.00262 ,   0.00268 ,   0.00262 ,
12.51,  0.000747,   0.000767,   0.000773,   0.000642,   0.000719,
22.24,  0.000154,   0.000125,   0.000131,   0.000144,   0.000150,
34.75,  1.33e-05,   1.26e-05,   1.47e-05,   1.27e-05,   1.30e-05,
50.04,  6.29e-07,   6.61e-07,   8.09e-07,   6.92e-07,   7.81e-07
])


#KMC rates
FWkmc12 = np.array([
8 ,  0.00431 ,   0.00416 ,   0.00420 ,   0.00422 ,   0.00449 ,
13,  0.00285 ,   0.00259 ,   0.00244 ,   0.00262 ,   0.00248 ,
17,  0.00194 ,   0.00178 ,   0.00205 ,   0.00184 ,   0.00214 ,
24,  0.000917,   0.00103 ,   0.000963,   0.00109 ,   0.000930,
28,  0.000730,   0.000793,   0.000719,   0.000712,   0.000730,
34,  0.000310,   0.000313,   0.000376,   0.000302,   0.000382,
38,  0.000242,   0.000262,   0.000233,   0.000232,   0.000208,
44,  0.000125,   0.000124,   0.000101,   9.95e-05,   0.000103,
50,  5.22e-05,   4.94e-05,   5.58e-05,   5.03e-05,   5.12e-05,
62,  8.85e-06,   9.93e-06,   9.98e-06,   1.13e-05,   9.00e-06,
72,  2.55e-06,   2.38e-06,   2.35e-06,   2.02e-06,   2.29e-06
])


#CLE rates
FWkcle12 = np.array([
5 ,  0.00550 ,   0.00483 ,   0.00481 ,   0.00544 ,   0.00526 ,
22,  0.000750,   0.000773,   0.000681,   0.000763,   0.000749,
50,  2.61e-05,   2.35e-05,   2.63e-05,   2.47e-05,   2.11e-05,
68,  1.73e-06,   2.07e-06,   1.65e-06,   1.52e-06,   1.92e-06,
78,  4.25e-07,   3.42e-07,   4.54e-07,   3.82e-07,   4.19e-07,
89,  8.31e-08,   1.10e-07,   8.89e-08,   9.23e-08,   8.93e-08
])


################## Backward ########################


#KMC rates
BWkmc30 = np.array([
8 ,   0.000162,   0.000161,   0.000153,   0.000169,   0.000158,
12,   7.98e-05,   7.33e-05,   7.57e-05,   7.22e-05,   7.12e-05,
17,   3.38e-05,   3.58e-05,   3.45e-05,   3.56e-05,   3.21e-05,
22,   1.27e-05,   1.29e-05,   1.13e-05,   1.21e-05,   1.29e-05,
32,   3.00e-06,   3.32e-06,   2.54e-06,   2.86e-06,   2.47e-06,
42,   5.67e-07,   6.60e-07,   4.31e-07,   5.13e-07,   5.24e-07
])


#CLE rates
BWkcle30 = np.array([
5.56 ,  0.00117 ,   0.00109 ,   0.00118 ,   0.00106 ,   0.00106 ,
12.51,  0.000259,   0.000285,   0.000225,   0.000220,   0.000287,
22.24,  4.65e-05,   4.78e-05,   4.83e-05,   4.80e-05,   5.02e-05,
34.75,  4.88e-06,   5.96e-06,   4.75e-06,   6.48e-06,   5.03e-06,
42.05,  1.58e-06,   1.71e-06,   1.82e-06,   1.80e-06,   1.47e-06,
50.04,  4.29e-07,   4.17e-07,   4.25e-07,   4.47e-07,   4.24e-07,
58.73,  8.69e-08,   8.75e-08,   8.28e-08,   8.02e-08,   8.41e-08,
68.11,  1.53e-08,   1.69e-08,   1.63e-08,   1.63e-08,   2.15e-08
])


#KMC rates
BWkmc12 = np.array([
5 ,  0.000247,   0.000278,   0.000267,   0.000241,   0.000270,
8 ,  0.000147,   0.000159,   0.000157,   0.000173,   0.000165,
12,  8.30e-05,   8.59e-05,   1.09e-04,   9.28e-05,   8.15e-05,
22,  2.14e-05,   2.29e-05,   2.67e-05,   2.26e-05,   2.67e-05,
28,  1.04e-05,   1.11e-05,   1.09e-05,   1.19e-05,   1.18e-05,
35,  4.81e-06,   4.15e-06,   4.83e-06,   4.47e-06,   4.70e-06,
42,  1.62e-06,   1.57e-06,   1.66e-06,   1.48e-06,   1.58e-06
])


#CLE rates
BWkcle12 = np.array([
5.56 ,  0.00128 ,   0.00141 ,   0.00135 ,   0.00141 ,   0.00135 ,
12.51,  0.000345,   0.000374,   0.000310,   0.000329,   0.000345,
22.24,  7.37e-05,   7.49e-05,   9.11e-05,   8.58e-05,   7.36e-05,
34.75,  1.38e-05,   1.38e-05,   1.18e-05,   1.38e-05,   1.39e-05,
50.04,  1.41e-06,   1.32e-06,   1.29e-06,   1.37e-06,   1.02e-06,
58.73,  3.24e-07,   3.03e-07,   3.41e-07,   3.54e-07,   3.20e-07,
68.11,  7.82e-08,   7.46e-08,   6.73e-08,   6.83e-08,   6.78e-08,
76.12,  2.08e-08,   2.31e-08,   2.20e-08,   2.06e-08,   1.71e-08
])


FWkmc12  = FWkmc12.reshape(-1,6)
FWkcle12 = FWkcle12.reshape(-1,6)
FWkmc60  = FWkmc60.reshape(-1,6)
FWkcle60 = FWkcle60.reshape(-1,6)

BWkmc12  = BWkmc12.reshape(-1,6)
BWkcle12 = BWkcle12.reshape(-1,6)
BWkmc30  = BWkmc30.reshape(-1,6)
BWkcle30 = BWkcle30.reshape(-1,6)


BWΩ_kmc12  = BWkmc12[:,0]
BWkmc_r12  = BWkmc12[:,1:]
BWkmc_m12  = np.mean(BWkmc_r12, axis=1)
BWkmc_s12  = np.std(BWkmc_r12 , axis=1)

BWΩ_cle12  = BWkcle12[:,0]
BWkcle_r12 = BWkcle12[:,1:]
BWkcle_m12 = np.mean(BWkcle_r12, axis=1)
BWkcle_s12 = np.std(BWkcle_r12 , axis=1)

BWΩ_kmc30  = BWkmc30[:,0]
BWkmc_r30  = BWkmc30[:,1:]
BWkmc_m30  = np.mean(BWkmc_r30, axis=1)
BWkmc_s30  = np.std(BWkmc_r30 , axis=1)

BWΩ_cle30  = BWkcle30[:,0]
BWkcle_r30 = BWkcle30[:,1:]
BWkcle_m30 = np.mean(BWkcle_r30, axis=1)
BWkcle_s30 = np.std(BWkcle_r30 , axis=1)


FWΩ_kmc12  = FWkmc12[:,0]
FWkmc_r12  = FWkmc12[:,1:]
FWkmc_m12  = np.mean(FWkmc_r12, axis=1)
FWkmc_s12  = np.std(FWkmc_r12 , axis=1)

FWΩ_cle12  = FWkcle12[:,0]
FWkcle_r12 = FWkcle12[:,1:]
FWkcle_m12 = np.mean(FWkcle_r12, axis=1)
FWkcle_s12 = np.std(FWkcle_r12 , axis=1)

FWΩ_kmc60  = FWkmc60[:,0]
FWkmc_r60  = FWkmc60[:,1:]
FWkmc_m60  = np.mean(FWkmc_r60, axis=1)
FWkmc_s60  = np.std(FWkmc_r60 , axis=1)

FWΩ_cle60  = FWkcle60[:,0]
FWkcle_r60 = FWkcle60[:,1:]
FWkcle_m60 = np.mean(FWkcle_r60, axis=1)
FWkcle_s60 = np.std(FWkcle_r60 , axis=1)


#Convert instanton rates between different noise strengths
def convert(ko,S,Oo,On,interface=True):
    if interface==True:
        kn = ko * np.exp(S*(Oo - On)) * np.sqrt(On/Oo) 
    else:
        kn = ko * np.exp(S*(Oo - On))
    return kn


FWk12  = 0.0072
FWSinst12 = 0.1574
FWom12    = 5.56

FWΩ_neqi12 = np.linspace(FWom12,89,100)
FWkarr12     = convert(FWk12,FWSinst12,FWom12,FWΩ_neqi12,interface=True)

### kappa 60 ###
FWk60     = 0.0031269437212794065
FWSinst60 = 0.1916118013245267
FWom60    = 5.56

FWΩ_neqi60 = np.linspace(FWom60,50,100)
FWkarr60   = convert(FWk60,FWSinst60,FWom60,FWΩ_neqi60,interface=False)


BWk12     = 0.0015374853947063107
BWSinst12 = 0.1731842686176764
BWom12    = 5.56

BWΩ_neqi12 = np.linspace(BWom12,85,100)
BWkarr12   = convert(BWk12,BWSinst12,BWom12,BWΩ_neqi12,interface=True)

### kappa 30 ###
BWk30     = 0.0014710439527867091
BWSinst30 = 0.1778348234040087
BWom30    = 5.56

BWΩ_neqi30 = np.linspace(BWom30,76,100)
BWkarr30   = convert(BWk30,BWSinst30,BWom30,BWΩ_neqi30,interface=False)


####################### Plotting  #################################
### Combine all into one plot
fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(34,6))

### FW, kappa=12 ###
ax1.scatter(FWΩ_kmc12, FWkmc_m12   , s=50 ,         color='k' , zorder=6, label='KMC')
ax1.scatter(FWΩ_kmc12, FWkmc_m12/4 , s=50 , facecolors='none', edgecolors='k', zorder=6, label='$k_\mathrm{KMC} \, 2 / L$')
ax1.plot(FWΩ_neqi12  , FWkarr12    , "-"  , lw=3.0, color='C2', zorder=4, label='NEQI')
ax1.plot(FWΩ_cle12   , FWkcle_m12  , "--" , lw=3.0, color='C3', zorder=5, label='CLE')
#
ax1.errorbar(FWΩ_kmc12, FWkmc_m12 , yerr=2*FWkmc_s12 , fmt='o', color='k',  markersize=6, elinewidth=2.5, capsize=3, zorder=5)
ax1.errorbar(FWΩ_cle12, FWkcle_m12, yerr=2*FWkcle_s12, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

### FW, kappa=60 ###
ax2.scatter(FWΩ_kmc60 , FWkmc_m60   ,        s=50, color='k', zorder=6, label='KMC')
ax2.scatter(FWΩ_kmc60 , FWkmc_m60/4 ,        s=50, facecolors='none', edgecolors='k', zorder=6, label='$k_\mathrm{KMC} \, 2 / L$')
ax2.plot(   FWΩ_neqi60, FWkarr60    , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')
ax2.plot(   FWΩ_cle60 , FWkcle_m60  , "--", lw=3.0, color='C3', zorder=5, label='CLE')
#
ax2.errorbar(FWΩ_kmc60, FWkmc_m60 , yerr=2*FWkmc_s60 , fmt='o', color='k',  markersize=6, elinewidth=2.5, capsize=3, zorder=5)
ax2.errorbar(FWΩ_cle60, FWkcle_m60, yerr=2*FWkcle_s60, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

### BW, kappa=12 ###
ax3.scatter(BWΩ_kmc12 , BWkmc_m12  , s=50 ,         color='k' , zorder=6, label='KMC')
ax3.scatter(BWΩ_kmc12 , 4*BWkmc_m12,        s=50 , facecolors='none', edgecolors='k', zorder=6, label='$k_\mathrm{KMC} \, L / 2$')
ax3.plot(BWΩ_neqi12   , BWkarr12   , "-"  , lw=3.0, color='C2', zorder=4, label='NEQI')
ax3.plot(BWΩ_cle12    , BWkcle_m12 , "--" , lw=3.0, color='C3', zorder=5, label='CLE')
#
ax3.errorbar(BWΩ_kmc12, BWkmc_m12 , yerr=2*BWkmc_s12 , fmt='o', color='k',  markersize=6, elinewidth=2.5, capsize=3, zorder=5)
ax3.errorbar(BWΩ_cle12, BWkcle_m12, yerr=2*BWkcle_s12, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)

### BW, kappa=30 ###
ax4.scatter(BWΩ_kmc30 , BWkmc_m30  ,        s=50, color='k', zorder=6, label='KMC')
ax4.scatter(BWΩ_kmc30 , 4*BWkmc_m30,        s=50 , facecolors='none', edgecolors='k', zorder=6, label='$k_\mathrm{KMC} \, L / 2$')
ax4.plot(   BWΩ_neqi30, BWkarr30   , "-" , lw=3.0, color='C2', zorder=4, label='NEQI')
ax4.plot(   BWΩ_cle30 , BWkcle_m30 , "--", lw=3.0, color='C3', zorder=5, label='CLE')
#
ax4.errorbar(BWΩ_kmc30, BWkmc_m30 , yerr=2*BWkmc_s30 , fmt='o', color='k',  markersize=6, elinewidth=2.5, capsize=3, zorder=5)
ax4.errorbar(BWΩ_cle30, BWkcle_m30, yerr=2*BWkcle_s30, fmt='o', color='C3', markersize=6, elinewidth=2.5, capsize=3, zorder=5)
###

ax1.set_xlabel("$\Omega$",size=24,labelpad=8)
ax2.set_xlabel("$\Omega$",size=24,labelpad=8)
ax3.set_xlabel("$\Omega$",size=24,labelpad=8)
ax4.set_xlabel("$\Omega$",size=24,labelpad=8)

ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')

ax1.set_ylabel("$k$",size=24,labelpad=12)
#ax4.legend()
leg = ax1.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.68,0.85), fancybox=True, shadow=True)
leg = ax2.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.68,0.85), fancybox=True, shadow=True)
leg = ax3.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.68,0.85), fancybox=True, shadow=True)
leg = ax4.legend(ncol=1,loc='upper left', bbox_to_anchor=(0.68,0.85), fancybox=True, shadow=True)

ax1.text(0.91,0.9,'(a)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.91,0.9,'(b)',transform=ax2.transAxes,fontsize=24)
ax3.text(0.91,0.9,'(c)',transform=ax3.transAxes,fontsize=24)
ax4.text(0.91,0.9,'(d)',transform=ax4.transAxes,fontsize=24)

ax1.xaxis.set_major_locator(MultipleLocator(20))
ax1.xaxis.set_minor_locator(MultipleLocator(10))
ax2.xaxis.set_major_locator(MultipleLocator(20))
ax2.xaxis.set_minor_locator(MultipleLocator(10))
ax3.xaxis.set_major_locator(MultipleLocator(20))
ax3.xaxis.set_minor_locator(MultipleLocator(10))
ax4.xaxis.set_major_locator(MultipleLocator(20))
ax4.xaxis.set_minor_locator(MultipleLocator(10))

plt.tight_layout()
fig.subplots_adjust(wspace=0.2)
plt.savefig("1DEnzyme_benchmarks.pdf", format='pdf', dpi=600)

plt.show()

