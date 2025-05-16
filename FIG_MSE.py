#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import freefield, freefield_derivatives
from utils import gridless_cmf, mle_async

from utils import grid3D, scm, generate_source, generate_noise, square_array
from scipy.optimize import linear_sum_assignment

from scipy.linalg import block_diag, svd
from scipy.spatial.distance import cdist

from algosasync import admm, agd, tnnradmm

from utils import CRB_unc_Nsources_async
from joblib import delayed, Parallel
import time

def eval_perf(XYZs, XYZest, P, Pest):
    
    C = cdist(XYZs, XYZest)
    
    i1, i2 = linear_sum_assignment(C)

    errXYZ =np.sum(((XYZs[i1, :] - XYZest[i2, :]))**2)
    errP =np.sum((P[i1] - Pest[i2])**2)

    return errXYZ, errP

# number of simulations
Nt = 100

# number of shifts between arrays
Np = 10

# tested methods
methods = ['A-COMET1', 'A-COMET2', 'A-COMET2+ML', 'TNNR', 'TNNR+ML', 'ADMM', 'ADMM+ML', 'AGD', 'AGD+ML']
Nm = len(methods)

# errors and CRBs
ex = np.zeros([Nt, Np, Nm])
ep = np.zeros([Nt, Np, Nm])
em = np.zeros([Nt, Np, 3])
empost = np.zeros([Nt, Np, Nm])

CRBasync_x = np.zeros([Np])
CRBasync_p = np.zeros([Np])


for npar in range(Np):
    # wavenumber
    k = 80

    # aperture of the array
    aperture = 0.5
    # number of sensors on a line/column
    N = 5
    step = aperture / (N-1)
    
    # shifts between the arrays
    delta = step*(npar+1)/2
    
    # size of the Psi matrix
    Npsi = int((N+(npar+1)/2)**2)
    
    A1 = square_array(aperture, N, center=[delta/2, delta/2, 0], axis='z')
    A2 = square_array(aperture, N, center=[delta/2, -delta/2, 0], axis='z')
    A3 = square_array(aperture, N, center=[-delta/2, +delta/2, 0], axis='z')
    A4 = square_array(aperture, N, center=[-delta/2, -delta/2, 0], axis='z')
        
    A0 = square_array(aperture+delta, N, center=[0, 0, 0], axis='z')
 
    Array = np.vstack([A1, A2, A3, A4])
    
    # source positions and powers
    XYZs1 = np.array([-0.44, -0.14, 3.43])
    XYZs2 = np.array([-0.1, 0.42, 4.17])
    XYZs3 = np.array([0.34, 0.41, 4.33])
    
    XYZs = np.vstack([XYZs1, XYZs2,XYZs3])
    
    Nsource = 3
    
    p1 = 1
    p2 = 0.5
    p3 = 0.5
    
    P = np.array([p1, p2, p3])
    
    # Number of snapshots and noise power
    Nsnaps = 256
    sigma2 = 0.1
    
      
    # ordering of the arrays
    idxasync = np.arange(4*(N**2))
    
        
    
    Arrayasync = Array[idxasync, :]
    
    # Green functions
    g = lambda x : freefield(Arrayasync, x, k)
    
    g1 = g(XYZs1)
    g2 = g(XYZs2)
    g3 = g(XYZs3)
    
    # theoretical covariance matrix
    
    Sigmath = p1 * np.outer(g1, g1.conj()) + p2 * np.outer(g2, g2.conj()) + p3 * np.outer(g3, g3.conj()) + sigma2 * np.eye(Array.shape[0])
    

    # grid for Psi matrix
    Xinit, diminit = grid3D([-1, -1, 3], [1, 1, 5], 0.1)
    Aadmm = g(Xinit)
    ua, sa, va = svd(Aadmm)
    
    box = np.array([[-1, -1, 3],[1, 1, 5]])
    
    
    Na = 4
    Ns = N**2
    Lseg = Nsnaps//Na
    
    gasync = []
    Sigmaasync = []
    aa = []
    
    Nsnapsest = [Nsnaps//Na] * Na

    # Green functions for each subarray
    for m in range(Na):
        aa.append(np.copy(Arrayasync[Ns*m:Ns*m+Ns, :]))
    
        gasync.append(lambda x, a=aa[m] : freefield(a, x, k))
    
      
    # same with derivatives
    gasyncd = []
    for m in range(Na):
            gasyncd.append(lambda x, a=aa[m] : freefield_derivatives(a, x, k))
    
    # CRBs
    
    CRBasync = CRB_unc_Nsources_async(Nsnapsest, XYZs, [p1, p2, p3], sigma2, gasyncd)
    
    CRBasync_p[npar] = np.sum(np.diag(CRBasync)[::4])
    CRBasync_x[npar] = np.sum(np.diag(CRBasync)[:-1]) - np.sum(np.diag(CRBasync)[::4])
    

# function to compute MSE for a given parameter and a given simulation
def result(npar, nt):
    k = 80

    aperture = 0.5
    N = 5
    step = aperture / (N-1)
    delta = step*(npar+1)/2
    
    Npsi = int((N+(npar+1)/2)**2)
    
    A1 = square_array(aperture, N, center=[delta/2, delta/2, 0], axis='z')
    A2 = square_array(aperture, N, center=[delta/2, -delta/2, 0], axis='z')
    A3 = square_array(aperture, N, center=[-delta/2, +delta/2, 0], axis='z')
    A4 = square_array(aperture, N, center=[-delta/2, -delta/2, 0], axis='z')
        
    A0 = square_array(aperture+delta, N, center=[0, 0, 0], axis='z')
 
    Array = np.vstack([A1, A2, A3, A4])
    
    
    XYZs1 = np.array([-0.44, -0.14, 3.43])
    XYZs2 = np.array([-0.1, 0.42, 4.17])
    XYZs3 = np.array([0.34, 0.41, 4.33])
    
    XYZs = np.vstack([XYZs1, XYZs2,XYZs3])
    
    Nsource = 3
    
    p1 = 1
    p2 = 0.5
    p3 = 0.5
    
    P = np.array([p1, p2, p3])
    
    Nsnaps = 256
    sigma2 = 0.1
    
    
    
    #%%
    # source model
    
    
    idxasync = np.arange(4*(N**2))
    
    Arrayasync = Array[idxasync, :]
    g = lambda x : freefield(Arrayasync, x, k)
    
    g1 = g(XYZs1)
    g2 = g(XYZs2)
    g3 = g(XYZs3)
    
    Sigmath = p1 * np.outer(g1, g1.conj()) + p2 * np.outer(g2, g2.conj()) + p3 * np.outer(g3, g3.conj()) + sigma2 * np.eye(Array.shape[0])
    

    
    Xinit, diminit = grid3D([-1, -1, 3], [1, 1, 5], 0.1)
    Aadmm = g(Xinit)
    ua, sa, va = svd(Aadmm)
    
    box = np.array([[-1, -1, 3],[1, 1, 5]])
    
    
    Na = 4
    Ns = N**2
    Lseg = Nsnaps//Na
    
    #%%
    
    Xinit, diminit = grid3D([-1, -1, 3], [1, 1, 5], 0.05)
        
    # data generation
    
    sig1 = generate_source(g(XYZs1), Nsnaps, p1)
    sig2 = generate_source(g(XYZs2), Nsnaps, p2) 
    sig3 = generate_source(g(XYZs3), Nsnaps, p3) 
    noise = generate_noise(Array.shape[0], Nsnaps, sigma2)
    
    sig = sig1 + sig2 + sig3 + noise
    
    # SCM/CSM
    Sigma = scm(sig)
 
    
    gasync = []
    Sigmaasync = []
    aa = []
    
    # local CSMs/SCMs
    for m in range(Na):
        aa.append(np.copy(Arrayasync[Ns*m:Ns*m+Ns, :]))
    
        gasync.append(lambda x, a=aa[m] : freefield(a, x, k))
    
        Sigma_loc = scm(sig[Ns*m: Ns*m+Ns, m*Lseg:m*Lseg+Lseg])
    
        Sigmaasync.append(Sigma_loc)
        
    D_meas = block_diag(*Sigmaasync)
    

    Psiadmm = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()
    Psiagd = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()
    
    lamadmm = 1e-4   
    lamagd = 1e-4  

    # completion methods
    Sadmm = admm(D_meas, Psiadmm, 1000, 2.6, 24.5/Array.shape[0], lamadmm)
    Sagd = agd(D_meas, 1000, lamagd, Psiagd, Nsource)
    Stnnr = tnnradmm(D_meas, Psiadmm, 100, 24.5/Array.shape[0], Nsource)
    
    em[nt, npar, 1] = np.sum(np.abs((Sigma-Sadmm)**2))
    em[nt, npar, 2] = np.sum(np.abs((Sigma-Sagd)**2))
    em[nt, npar, 0] = np.sum(np.abs((Sigma-Stnnr)**2))

    
    Nsnapsest = [Nsnaps//Na] * Na
    
    # gridless estimations, COMET1 and MLE

    Xcmf, Pcmf, sigma2cmf = gridless_cmf(Sigmaasync, Nsnapsest,  gasync, Xinit, Nsource, box, mode="comet2")
    Xmle, Pmle, sigma2mle = mle_async(Sigmaasync, Nsnapsest, gasync, Xcmf, Pcmf, sigma2cmf, box)
    
    Xcmf1, Pcmf1, sigma2cmf1 = gridless_cmf(Sigmaasync, Nsnapsest,  gasync, Xinit, Nsource, box, mode="comet1")

    Xadmm, Padmm, sigma2admm = gridless_cmf([Sadmm], [Nsnaps], [g], Xinit, Nsource, box, mode="comet2")
    Xadmmmle, Padmmmle, sigma2admmmle = mle_async([Sadmm], [Nsnaps], [g], Xadmm, Padmm, sigma2admm, box)

    Xtnnr, Ptnnr, sigma2tnnr = gridless_cmf([Stnnr], [Nsnaps], [g], Xinit, Nsource, box, mode="comet2")
    Xtnnrmle, Ptnnrmle, sigma2tnnrmle = mle_async([Stnnr], [Nsnaps], [g], Xtnnr, Ptnnr, sigma2tnnr, box)
  
    Xagd, Pagd, sigma2agd = gridless_cmf([Sagd], [Nsnaps], [g], Xinit, Nsource, box, mode="comet2")
    Xagdmle, Pagdmle, sigma2agdmle = mle_async([Sagd], [Nsnaps], [g], Xagd, Pagd, sigma2agd, box)
  
    Xests = (Xcmf1, Xcmf, Xmle, Xtnnr, Xtnnrmle, Xadmm, Xadmmmle, Xagd, Xagdmle)
    Pests = (Pcmf1, Pcmf, Pmle, Ptnnr, Ptnnrmle, Padmm, Padmmmle, Pagd, Pagdmle)
    Sests = (sigma2cmf1, sigma2cmf, sigma2mle, sigma2tnnr, sigma2tnnrmle, sigma2admm, sigma2admmmle, sigma2agd, sigma2agdmle)

    exloc = np.zeros(Nm)
    eploc = np.zeros(Nm)
    empostloc = np.zeros(Nm)

    # computation of the MSE and matrix completion error
    for nm in range(Nm):
        
        Xest = Xests[nm]
        Pest = Pests[nm]
        sigma2est = Sests[nm]
        
        
        exloc[nm], eploc[nm] = eval_perf(XYZs, Xests[nm], P, Pests[nm])

        gest = g(Xest)
        
        Sigmaest = gest @ np.diag(Pest) @ gest.T.conj() + sigma2est * np.eye(Array.shape[0])
        
        empostloc[nm] = np.sum(np.abs((Sigmaest-Sigmath)**2))
 
    return npar, nt, exloc, eploc, empostloc

#%% running the simulations with joblib
T = time.time()
Z = Parallel(n_jobs=16)(delayed(result)(i, j) for i in range(Np) for j in range(Nt))
TT = time.time() - T

#%% untangling the errors

for npar, nt, exloc, eploc, empostloc in Z:
    ex[nt, npar, :] = exloc
    ep[nt, npar, :] = eploc
    empost[nt, npar, :] = empostloc


#%% saving and plotting

np.save("ex", ex)
np.save("ep", ep)
np.save("empost", empost)

delta = (np.arange(Np)+1)/2

EX = np.mean(ex, axis=0)
EP = np.mean(ep, axis=0)
EMPOST = np.mean(empost, axis=0)



plt.figure()

plt.semilogy(delta, EX[:, 1], '-+', label=methods[1])
plt.semilogy(delta, EX[:, 2], '-x', label=methods[2])
plt.semilogy(delta, np.min(EX[:, 3:], axis=1), "-o", label="best compl.")
plt.semilogy(delta, CRBasync_x, label="CRB", linewidth=2)

plt.xlabel("$\Delta$")
plt.ylabel("MSE ($m^2$)")

plt.legend()

plt.figure()

plt.semilogy(delta, EP[:, 1], '-+', label=methods[1])
plt.semilogy(delta, EP[:, 2], '-x', label=methods[2])
plt.semilogy(delta, np.min(EP[:, 3:], axis=1), "-o", label="best compl.")
plt.semilogy(delta, CRBasync_p, label="CRB", linewidth=2)

plt.ylabel("MSE ($Pa^4$)")
plt.xlabel("$\Delta$")

plt.legend()


plt.figure()

plt.semilogy(delta, EMPOST[:, 1], '-+', label=methods[1])
plt.semilogy(delta, EMPOST[:, 2], '-x', label=methods[2])
plt.semilogy(delta, np.min(EMPOST[:, 3:], axis=1), "-o", label="best compl.")

plt.ylabel("MSE ($Pa^4$)")
plt.xlabel("$\Delta$")

plt.legend()

