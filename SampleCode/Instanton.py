#!/usr/bin/env python
# coding: utf-8

""" Compute nonequilibrium instanton rate constant for a Schlögl model extended along one spatial dimension"""                                                 
__author__ = 'Eric Heller'
import numpy as np
import scipy as sp
import scipy.optimize as optimize
from scipy.linalg import cholesky_banded, solveh_banded
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


# Optional tex-based plotting options
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
N      = 512     # Beads
# System specification
dof    = 40      # Number of voxels
dx     = 1.0     # Voxel length
length = dof*dx
# Reaction network
kp1    = 0.8 
kp2    = 3.1
km1    = 2.9
km2    = 1
# Diffusion strength
kappa  = 4
# Noise
Ω      = 1e+2     # Typical number of molecules per compartment


### Deterministic drift and its derivatives ### 
def force(x):
    f  = kp2*x**2 - km2*x**3 - km1*x + kp1 
    f -= kappa * (2*x - np.roll(x, 1) - np.roll(x, -1)) 
    return f

def dforcedx(x):
    df        = np.diag(2*kp2*x - 3*km2*x**2 - km1 -2*kappa) 
    offdiag   = kappa*np.ones(dof-1)
    df       += np.diag(offdiag,k=1)
    df       += np.diag(offdiag,k=-1)
    df[0,-1] += kappa  
    df[-1,0] += kappa
    return df

def d2forcedx2(x):
    d2f = 2*kp2 - 6*km2*x
    return d2f

### Reaction noise matrix and its derivatives, using its diagonal structure for dimensionality reduction ###
def Dmat_diag(x):
    tmp        = kp2*x**2 + km2*x**3 + km1*x + kp1  
    return tmp

def Ddet(x):
    matD = Dmat_diag(x)
    Ddet = np.prod(matD)
    return Ddet

def Dinv_diag(x):
    matD = Dmat_diag(x)
    return 1/matD

def Dinvdx(x):
    maindiag = -(km1 + 2*kp2*x + 3*km2*x**2) / (kp1 + km1*x + kp2*x**2 + km2*x**3)**2
    Df = np.diag(maindiag)
    return Df

def Dinvdx2(x):
    den               = kp1 + km1*x + kp2*x**2 + km2*x**3
    dden              = km1 + 2*kp2*x + 3*km2*x**2
    d2den             = 2*kp2 + 6*km2*x
    maindiag          = 2*dden**2 / den**3 - d2den / den**2
    Df                = np.zeros((dof, dof, dof))
    idx               = np.arange(dof)
    Df[idx, idx, idx] = maindiag
    return Df 


# Path action and its derivatives
def functionalx(x,xa,xb):    # xa and xb are the fixed ends
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    S = 0
    for indn in range(len(xp)-1):
        Dinv = Dinv_diag(xp[indn])
        pmat = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        S += 0.5*dx*tau*np.dot(pmat,Dinv*pmat)
    return S 

def functionalarray(x,xa,xb):    # xa and xb are the fixed ends
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    S = np.zeros(N)
    for indn in range(len(xp)-1):
        Dinv = Dinv_diag(xp[indn])
        pmat = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        S[indn] = 0.5*dx*tau*np.dot(pmat,Dinv*pmat)
    return S 

def dSdx(x,xa,xb):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    part1 = np.zeros((N,dof))
    part2 = np.zeros((N,dof))
    for indn in range(len(xp)-1):
        Dinv  = Dinv_diag(xp[indn])
        pDarr = Dinvdx(xp[indn])
        pmat  = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat = -np.diag(np.ones(dof))/tau - dforcedx(xp[indn])
        part1[indn] = dx*tau*np.matmul(pmat*Dinv,ppmat) + 0.5*tau*np.matmul(pmat*pDarr.T,pmat)
        part2[indn] = dx*pmat*Dinv
    dSdx = part1[1:] + part2[:-1] 
    return dSdx.ravel()

### One end open ###
def dSdx_one(x,xa):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),x],axis=0)
    part1 = np.zeros((len(xp),dof))
    part2 = np.zeros((len(xp),dof))
    for indn in range(len(xp)-1):
        Dinv  = Dinv_diag(xp[indn])
        pDarr = Dinvdx(xp[indn])
        pmat  = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat = -np.diag(np.ones(dof))/tau - dforcedx(xp[indn])
        part1[indn] = dx*tau*np.matmul(pmat*Dinv,ppmat) + 0.5*tau*np.matmul(pmat*pDarr.T,pmat)
        part2[indn] = dx*pmat*Dinv
    dSdx = part1[1:] + part2[:-1] 
    return dSdx.ravel()

# Banded second-derivative matrix
def d2Sdx2_bandeddirect(x,xa,xb):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    hessb = np.zeros((2*dof,(N-1)*dof))
    for indn in range(1,len(xp)-1):
        Dinv     = Dinv_diag(xp[indn])
        pDarr    = Dinvdx(xp[indn])
        ppDarr   = Dinvdx2(xp[indn])
        pmat     = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat    = np.diag(np.ones(dof))/tau + dforcedx(xp[indn])
        pppmat   = d2forcedx2(xp[indn])
        dp       = np.matmul(ppmat,pmat.reshape(-1,1)*Dinvdx(xp[indn]))

        diag1    = dx * tau * (-np.diag(Dinv*pppmat*pmat) + np.matmul(ppmat,Dinv.reshape(-1,1)*ppmat) - dp - dp.T + 0.5*np.einsum('jkl,j->kl',ppDarr*pmat.reshape(-1,1,1),pmat))
        diag2    = dx/tau*np.diag(Dinv_diag(xp[indn-1])) 
        diag     = diag1 + diag2
        offdiag  = -ppmat.T * Dinv + pDarr.T * pmat.reshape(1,-1) 
        offdiag *= dx

        diagB    = np.zeros((dof,dof))
        offB     = np.zeros((2*dof-1,dof))
        for indf in range(dof):
            diagB[-(indf+1),indf:]  = np.diag(diag,k=indf)
            offB[-(indf+1),:indf+1] = np.diag(offdiag.T,k=dof-indf-1)
            offB[indf,-(indf+1):]   = np.diag(offdiag,k=dof-indf-1)
        hessb[-dof:,(indn-1)*dof:indn*dof] += diagB
        if indn < len(xp)-2:
            hessb[-(2*dof+1):-1,indn*dof:(indn+1)*dof] += offB
    return hessb

def d2Sdx2_one_bandeddirect(x,xa):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),x],axis=0)
    hessb = np.zeros((2*dof,N*dof))
    for indn in range(1,len(xp)-1):
        Dinv     = Dinv_diag(xp[indn])
        pDarr    = Dinvdx(xp[indn])
        ppDarr   = Dinvdx2(xp[indn])
        pmat     = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat    = np.diag(np.ones(dof))/tau + dforcedx(xp[indn])
        pppmat   = d2forcedx2(xp[indn])
        dp       = np.matmul(ppmat,pmat.reshape(-1,1)*Dinvdx(xp[indn]))

        diag1    = dx*tau * (-np.diag(Dinv*pppmat*pmat) + np.matmul(ppmat,Dinv.reshape(-1,1)*ppmat) - dp - dp.T + 0.5*np.einsum('jkl,j->kl',ppDarr*pmat.reshape(-1,1,1),pmat))
        diag2    = dx/tau*np.diag(Dinv_diag(xp[indn-1])) 
        diag     = diag1 + diag2
        offdiag  = -ppmat.T * Dinv + pDarr.T * pmat.reshape(1,-1) 
        offdiag *= dx

        diagB = np.zeros((dof,dof))
        offB  = np.zeros((2*dof-1,dof))
        for indf in range(dof):
            diagB[-(indf+1),indf:]  = np.diag(diag,k=indf)
            offB[-(indf+1),:indf+1] = np.diag(offdiag.T,k=dof-indf-1)
            offB[indf,-(indf+1):]   = np.diag(offdiag,k=dof-indf-1)
        hessb[-dof:,(indn-1)*dof:indn*dof] += diagB
        if indn < len(xp)-1:
            hessb[-(2*dof+1):-1,indn*dof:(indn+1)*dof] += offB
    # Final bead
    diagB = np.zeros((dof,dof))
    diag = dx/tau*np.diag(Dinv_diag(xp[N-1]))
    for indf in range(dof):
        diagB[-(indf+1),indf:]  = np.diag(diag,k=indf)
    hessb[-dof:,(N-1)*dof:N*dof] += diagB
    return hessb


### Locate relevant fixed points of the drift ###
# Reactant
guess = 0.5*np.ones(dof)
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
minR = sol.x
jacR = dforcedx(minR)
print("Jacobian EV at reactant minimum",np.linalg.eigvals(jacR))

# Product
guess = 1.6*np.ones(dof)
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
minP = sol.x
jacP = dforcedx(minP)
print("Jacobian EV at product minimum",np.linalg.eigvals(jacP))

# TS
guess = np.genfromtxt("TS.txt")
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-30, method='hybr')
TS = sol.x
jacTS = dforcedx(TS)
print("Jacobian EV at TS",np.sort(np.linalg.eigvals(jacTS)))


# Plot fixed points of the drift
fig,ax = plt.subplots(figsize=(8.5,6))
xgrid = np.linspace(0,length,dof)
plt.plot(xgrid, minR, lw=3.0, label='R' )
plt.plot(xgrid, minP, lw=3.0, label='P' )
plt.plot(xgrid, TS  , lw=3.0, label='TS')
plt.xlabel(r"$x$",size=24,labelpad=8)
plt.ylabel(r"$\rho$",size=24,labelpad=12)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.3))
ax.yaxis.set_minor_locator(MultipleLocator(0.15))

plt.xlim(np.min(xgrid),np.max(xgrid))

plt.legend()
plt.tight_layout()
plt.show()


### Save translational mode ###
eigTS, evecTS = np.linalg.eig(-jacTS)
translindex = np.argmin(np.abs(eigTS))
translmode = evecTS[:,translindex] 


### Create initial guess for trajectory ###
def instguess(N,minR,minP,TS):      # N number of Beads, y0 TS-coordinate
    yR = np.linspace(minR,TS,N//2+1) 
    yP = np.linspace(TS,minP,N//2+1)
    y = np.concatenate([yR,yP[1:]],axis=0)
    return y

endpoint = TS    # Right option for the activation path from reactant to TS
#endpoint = minP # Right option for the full optimal transition path from reactant to product
iguessx = instguess(N-2,minR,endpoint,minR+0.5*(endpoint-minR))
iguessx = np.genfromtxt("xinitNewton.txt").reshape(-1,dof)
########################
# Total time and time step
T = 50
tau = T / N


# Functions for instanton optimization
def potx(x):
    S = functionalx(x,minR,endpoint)
    return S

def Sgrad(x):
    dS = dSdx(x,minR,endpoint)
    return dS

def Shess(x):
    d2S = d2Sdx2_bandeddirect(x,minR,endpoint)
    return d2S


# Instanton optimization using SciPy's trust-krylov method. This is a general-purpose second-order optimizer.
# More specialized algorithms for instanton or path optimization are also available.

# Matrix-vector product for a symmetric banded matrix in LAPACK upper-band storage.
def sym_banded_matvec(ab, x):
    u, n = ab.shape[0] - 1, ab.shape[1]
    y    = np.zeros_like(x)
    y   += ab[u] * x
    for k in range(1, u + 1):
        diag    = ab[u - k, k:]
        y[k:]  += diag * x[:-k]
        y[:-k] += diag * x[k:]
    return y

# Cache the Hessian so it is not rebuilt in every Krylov subiteration.
class HessPCache:
    def __init__(self, hessb_fun):
        self.hessb_fun = hessb_fun
        self.x_cached  = None
        self.ab_cached = None
        self.n_build   = 0
        self.n_matvec  = 0

    def hessp(self, x, p):
        self.n_matvec += 1
        if self.x_cached is None or not np.array_equal(x, self.x_cached):
            self.x_cached  = x.copy()
            self.ab_cached = self.hessb_fun(x)
            self.n_build  += 1
        return sym_banded_matvec(self.ab_cached, p)

cache = HessPCache(Shess)

# Callback function to print one line of diagnostics per outer iteration.
class OptimLogger:
    def __init__(self, fun, grad):
        self.fun    = fun
        self.grad   = grad
        self.iter   = 0
        self.prev_x = None

        print("\n iter      S(x)           |g|_inf        |g|_2        step")

    def __call__(self, xk):
        S = self.fun(xk)
        g = self.grad(xk)

        ginf = np.max(np.abs(g))  # Max gradient component
        g2   = np.linalg.norm(g)  # Gradient norm

        if self.prev_x is None:
            step = 0.0
        else:
            step = np.linalg.norm(xk - self.prev_x)

        print(f"{self.iter:5d}  {S: .6e}  {ginf: .3e}  {g2: .3e}  {step: .3e}")

        self.prev_x = xk.copy()
        self.iter += 1
        
logger = OptimLogger(potx, Sgrad)

# Optimization
sol   = optimize.minimize(fun=potx, x0=iguessx.ravel(), jac=Sgrad, hessp=cache.hessp, method='trust-krylov', callback=logger,
                        options={'gtol': 1e-6,'maxiter': 10, 'disp': False})

instx = sol.x.reshape(N-1,dof)

# Print results
print("\n=== Optimization summary ===")
print(f"Success:        {sol.success}")
print(f"Message:        {sol.message}")
print(f"Iterations:     {sol.nit}")
print(f"Func evals:     {sol.nfev}")
print(f"Grad evals:     {sol.njev}")
print(f"Final S:        {sol.fun:.6e}")
print()

action = potx(instx)
g      = Sgrad(instx)
print("Action",action)
print(f"Final |g|_inf:  {np.max(np.abs(g)):.6e}")
print(f"Final |g|_2:    {np.linalg.norm(g):.6e}")
print(f"Hessian builds:  {cache.n_build}")
print(f"Hessp calls:     {cache.n_matvec}")

# Save result
np.savetxt("xinitNewton.txt",instx)

# Create instanton paths that include the endpoints
insto = np.concatenate([instx,endpoint.reshape(-1,dof)],axis=0)
instp = np.concatenate([np.concatenate([minR.reshape(1,dof),instx.reshape(-1,dof)],axis=0),endpoint.reshape(1,dof)],axis=0)

# Plot action change along the path
Sarray = functionalarray(instx,minR,endpoint)
tplot = np.linspace(0.0,T,N+1)

fig,ax = plt.subplots(figsize=(8.5,6))
plt.xlim(np.min(tplot),np.max(tplot))
plt.plot(tplot[1:],Sarray,lw=3.0,color='C0',zorder=3)
ax.set_xlabel("$t$",size=24,labelpad=8)
ax.set_ylabel("$\mathrm{d}S$"  ,size=24,labelpad=12)
plt.tight_layout()
plt.show()


# Plot evolution of the concentration profile in time
fig,ax = plt.subplots(figsize=(8.5,6))

tplot = np.linspace(0.0,T,N+1)
xplot = np.linspace(0.0,length,dof)
TIME, X = np.meshgrid(tplot,xplot)
ax.set_xlabel("$t$",fontsize=24, labelpad=8)
ax.set_ylabel("$x$",fontsize=24, labelpad=12)

cm = ax.contourf(TIME,X,instp.T,levels=20)

cbar = fig.colorbar(cm)
ticks = np.linspace(instp.min(), instp.max(), 6)
cbar.set_ticks(ticks)
cbar.set_label(r"$\rho$",size=24,labelpad=12)

plt.xlim(np.min(tplot),np.max(tplot))
plt.ylim(np.min(xplot),np.max(xplot))

plt.tight_layout()
plt.show()


# Function to compute the pseudo-determinant of the second-derivative matrix of the action 
# with the reaction coordinate and zero-modes from temporal and spatial translations removed
# without ever densifying the banded matrix
def logdet_prime_banded(hess_band, ttrans, p_end, spatial_end_modes, tau_scalar, dx,
                        block_size, lower=False, check_finite=False):
    ab     = np.asarray(hess_band, dtype=float)
    ttrans = np.asarray(ttrans, dtype=float)

    n      = ab.shape[1]
    u      = ab.shape[0] - 1
    d      = int(block_size)

    if ttrans.shape != (n,):
        raise ValueError(f"ttrans must have shape ({n},)")
    if not (d >= 1 and n >= 2 * d):
        raise ValueError("Need block_size >= 1 and at least two blocks.")
    if tau_scalar <= 0.0 or dx <= 0.0:
        raise ValueError("tau_scalar and dx must be positive.")

    last_start = n - d
    prev_start = n - 2 * d

    SM = np.asarray(spatial_end_modes if spatial_end_modes is not None else [], dtype=float)
    if SM.size == 0:
        SM = np.empty((0, d))
    elif SM.ndim == 1:
        if SM.shape[0] != d:
            raise ValueError(f"spatial_end_modes must have length {d} or shape (f,{d})")
        SM = SM[None, :]
    elif SM.ndim != 2 or SM.shape[1] != d:
        raise ValueError(f"spatial_end_modes must have shape (f,{d})")

    f = SM.shape[0]
    r = 1 + f
    if d <= r:
        raise ValueError(f"Need block_size > 1 + number of spatial modes; got d={d}, f={f}")

    p_end = np.asarray(p_end, dtype=float)
    if p_end.shape[0] == n:
        p_blk = p_end[last_start:]
    elif p_end.shape[0] == d:
        p_blk = p_end
    else:
        raise ValueError(f"p_end must have length {d} or {n}")

    W = np.empty((d, r))
    W[:, 0] = p_blk
    if f:
        W[:, 1:] = SM.T

    Q0, R0 = np.linalg.qr(W, mode="complete")
    if np.any(np.abs(np.diag(R0[:r, :r])) < 1e-12 * max(1.0, np.max(np.abs(R0[:r, :r])))):
        raise ValueError("Local modes are linearly dependent or near-zero.")

    Q = np.hstack([Q0[:, r:], Q0[:, :r]])
    d_keep = d - r

    def get_band(i, j):
        if lower:
            if i < j:
                i, j = j, i
            k = i - j
            return 0.0 if k > u else ab[k, j]
        else:
            if i > j:
                i, j = j, i
            k = u + i - j
            return 0.0 if (k < 0 or k > u) else ab[k, j]

    def set_band(abX, uX, i, j, val):
        if lower:
            if i < j:
                i, j = j, i
            k = i - j
            if k <= uX:
                abX[k, j] = val
        else:
            if i > j:
                i, j = j, i
            k = uX + i - j
            if 0 <= k <= uX:
                abX[k, j] = val

    D      = np.array([[get_band(last_start + a, last_start + b) for b in range(d)] for a in range(d)])
    K      = np.array([[get_band(prev_start + a, last_start + b) for b in range(d)] for a in range(d)])

    Dp     = Q.T @ D @ Q
    Kp     = K @ Q

    A      = Dp[:d_keep, :d_keep]
    C      = Kp[:, :d_keep]

    n_red  = n - r
    u_red  = min(u, n_red - 1)
    ab_red = ab[:u_red + 1, :n_red].copy() if lower else ab[(u - u_red):(u + 1), :n_red].copy()
    u_red  = ab_red.shape[0] - 1

    for a in range(d_keep):
        ia = last_start + a
        for b in range(d_keep):
            set_band(ab_red, u_red, ia, last_start + b, A[a, b])

    for a in range(d):
        ia = prev_start + a
        for b in range(d_keep):
            set_band(ab_red, u_red, ia, last_start + b, C[a, b])

    t_last_p    = Q.T @ ttrans[last_start:]
    t_red       = np.concatenate([ttrans[:last_start], t_last_p[:d_keep]])

    Ufac        = cholesky_banded(ab_red, lower=lower, overwrite_ab=False, check_finite=check_finite)
    diag        = Ufac[0] if lower else Ufac[-1]
    logdetH_red = 2.0 * np.sum(np.log(diag))

    yt = solveh_banded(ab_red, t_red, lower=lower, overwrite_ab=False, check_finite=check_finite)
    s  = float(np.dot(t_red, yt))
    if s <= 0.0:
        raise ValueError("t^T H_red^{-1} t is not positive.")

    return logdetH_red + np.log(s) + (n_red - 1) * np.log(tau_scalar / dx)


### Determinant-based NEQI rate ###
def rateB_det(Ω,interface=True):
    gradO = dSdx_one(insto,minR)
    hessO = d2Sdx2_one_bandeddirect(insto,minR)
    # Time-translation mode 
    ttransp = (instp[1:] - instp[:-1]) / tau    # shape (N, dof)     
    ttransp  = ttransp.ravel()
    ttransp /= np.linalg.norm(ttransp)
    # Time translation Jacobian
    BN  = np.sum((instp[1:] - instp[:-1])**2) / tau
    # Spatial translations at endpoint
    if interface ==True:
        print("Interface-forming mechanism specified")
        modes = [translmode]
    else:
        print("Homogeneous mechanism specified")
        modes = []
    # Momentum
    pf    = gradO[-dof:]
    pvec  = pf / np.linalg.norm(pf)
    momentum = np.linalg.norm(pf)

    # Fluctuation determinant
    logdetb = logdet_prime_banded(hessO, ttransp, pvec, modes, tau, dx, block_size = dof)

    # Path measure
    Ddetarr = np.zeros(N)
    for indn in range(N):
        Ddetarr[indn] = Ddet(instp[indn])
    detI = -0.5*logdetb - 0.5*np.sum(np.log(Ddetarr))    
    detI = np.exp(detI)
   
    # Combine terms into prefactor
    pre = 1 / (2*np.pi*tau) * detI * np.sqrt(BN*dx**2/(tau*momentum**2)) 
    # Spatial translation Jacobian
    if interface ==True:
        ba   = kappa*np.sum(((np.roll(TS, -1) - TS)/dx)**2)
        Ba   = np.sqrt(ba*Ω*dx)/np.sqrt(2*np.pi*tau*kappa)
        pre *= Ba * length
    # Rate constant
    k = pre * np.exp(-Ω*action)
    return k


### Eigenvalue-based NEQI rate ###
def rateB_eig(Ω,interface=True):
    gradO    = dSdx_one(insto,minR)
    hessB    = d2Sdx2_one_bandeddirect(insto,minR)
    BN       = np.sum((instp[1:] - instp[:-1])**2) / tau
    
    # Momentum
    pf       = gradO[-dof:]
    momentum = np.linalg.norm(pf)
    pvec     = pf / np.linalg.norm(pf)
    pblock   = np.identity(dof)  - np.outer(pvec,pvec) 
    if interface==True:
        print("Interface-forming mechanism specified")
        pblock -= np.outer(translmode,translmode)
    else:
        print("Homogeneous mechanism specified")
    
    # Project out reaction coordinate and spatial translation at endpoint for interface-forming mechanisms
    Dblock  = np.zeros((dof,dof))
    Oblock1 = np.zeros((dof,dof))
    for indf in range(dof):
        Dblock  += np.diag(hessB[-(indf+1),-(dof-indf):],k=indf)
        Oblock1 += np.diag(hessB[-(dof+indf+1),-(dof-indf):],k=indf)
    for indf in range(dof-1):
        Oblock1 += np.diag(hessB[-(indf+2),-dof:-dof+indf+1],k=-dof+1+indf)
    Dblock  = Dblock+Dblock.T - np.diag(np.diag(Dblock))
    Oblock2 = Oblock1.T
    
    diagP = np.matmul(pblock.T,np.matmul(Dblock,pblock))
    off1 = np.matmul(Oblock1,pblock)
    off2 = np.matmul(pblock.T,Oblock2)
    for indf in range(dof):
        hessB[-(indf+1),-(dof-indf):] = np.diag(diagP,k=indf)
        hessB[-(dof+indf+1),-(dof-indf):] = np.diag(off1,k=indf)
        if indf != 0:
            hessB[-(indf+1),-dof:-dof+indf] = np.diag(off1.T,k=dof-indf)

    # Compute eigenvalues
    eigI = sp.linalg.eigvals_banded(tau*hessB/dx)
    print("First 20 eigenvalues after projection:",eigI[:20])
    # Remove zero eigenvalues
    if interface==True:
        eigI = eigI[3:] 
    else:
        eigI = eigI[2:]
    # Path measure
    Ddetarr = np.zeros(N)
    for indn in range(N):
        Ddetarr[indn] = Ddet(instp[indn])
    detI = -0.5*np.sum(np.log(eigI)) - 0.5*np.sum(np.log(Ddetarr))
    detI = np.exp(detI) 
    
    pre = 1 / (2*np.pi*tau) * detI * np.sqrt(BN*dx**2/(tau*momentum**2))
    # 0-mode Jacobian
    if interface ==True:
        ba   = kappa*np.sum(((np.roll(TS, -1) - TS)/dx)**2)
        Ba   = np.sqrt(ba*Ω*dx)/np.sqrt(2*np.pi*tau*kappa)
        pre *= Ba * length

    k = pre * np.exp(-Ω * action)
    return k


# Print results
print("Instanton action",action)
print()
knumD = rateB_det(Ω, interface=True)
print("Determinant rate", knumD)

# Uncomment for comparison to eigenvalue-based rate (longer calculation time)
#knumE = rateB_eig(Ω, interface=True) 
#print("Eigenvalue rate ", knumE)

