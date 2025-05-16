#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import freefield1D
#from acosolo.gridless import gridless_cmf, gridless_cond
from utils import gridless_cmf, mle_async

from utils import grid3D, scm, generate_source, generate_noise, square_array
from scipy.optimize import linear_sum_assignment

from scipy.linalg import block_diag, svd
from scipy.spatial.distance import cdist

from algosasync import bhmc, admm, apgl, agd, tnnradmm

import tqdm


def eval_perf(XYZs, XYZest, P, Pest):
    
    C = cdist(XYZs, XYZest)
    
    i1, i2 = linear_sum_assignment(C)

    errXYZ =np.sum(((XYZs[i1, :] - XYZest[i2, :]))**2)
    errP =np.sum((P[i1] - Pest[i2])**2)

    return errXYZ, errP


Nt = 1
Np = 60
Z = 4

K = 10+np.arange(Np)*2

delta_source = 0.2
Y = -0.24

Nsource = 2



XXcmf = np.zeros([Nsource, Np])

XXagd = np.zeros([Nsource, Np])

XXmle = np.zeros([Nsource, Np])

XXagdmle = np.zeros([Nsource, Np])

PPcmf = np.zeros([Nsource, Np])

PPagd = np.zeros([Nsource, Np])

PPmle = np.zeros([Nsource, Np])

PPagdmle = np.zeros([Nsource, Np])
 

for npar in tqdm.tqdm(range(Np)):
    k = K[npar]
     
    npar0 = 5
    
    aperture = 0.5
    N = 5
    step = aperture / (N-1)
    delta = step*(npar0+1)/2
    
    Npsi = int((N+(npar0+1)/2)**2)
    
    A1 = square_array(aperture, N, center=[delta/2, delta/2, 0], axis='z')
    A2 = square_array(aperture, N, center=[delta/2, -delta/2, 0], axis='z')
    A3 = square_array(aperture, N, center=[-delta/2, +delta/2, 0], axis='z')
    A4 = square_array(aperture, N, center=[-delta/2, -delta/2, 0], axis='z')
    
    
    A0 = square_array(aperture+delta, N, center=[0, 0, 0], axis='z')
    
    
    
    
    
    
    
    Array = np.vstack([A1, A2, A3, A4])
    
    
    XYZs1 = np.array([-delta_source/2+0.238])
    XYZs2 = np.array([delta_source/2+0.238])
    
    XYZs = np.vstack([XYZs1, XYZs2])
    
    
    p1 = 1
    p2 = 1
    
    P = np.array([p1, p2])
    
    Nsnaps = 256
    sigma2 = 0.1
    

    
    idxasync = np.arange(4*(N**2))
    

    Arrayasync = Array[idxasync, :]
    g = lambda x : freefield1D(Arrayasync, x, Y, Z, k)
    
    g1 = g(XYZs1)
    g2 = g(XYZs2)
    
    Sigmath = p1 * np.outer(g1, g1.conj()) + p2 * np.outer(g2, g2.conj()) + sigma2 * np.eye(Array.shape[0])
    
    g0 = lambda x : freefield1D(A0, x, Y, Z, k)
    
    
    Xinit, diminit = grid3D([-1, -1, 4], [1, 1, 4], 0.1)
    Aadmm = g(Xinit)
    ua, sa, va = svd(Aadmm)
    
    box = np.array([[-1],[1]])
    
    
    Na = 4
    Ns = N**2
    
    Lseg = Nsnaps//Na
    
    #%%
    
    Xinit, diminit = grid3D([-1, Y, 4], [1, Y, 4], 0.01)
    Xinit = Xinit[:, 0:1]
    

    
    for nt in range(Nt):
        
        sig1 = generate_source(g(XYZs1), Nsnaps, p1)
        sig2 = generate_source(g(XYZs2), Nsnaps, p2) 
        noise = generate_noise(Array.shape[0], Nsnaps, sigma2)
        
        sig = sig1 + sig2 + noise
        
            
        Sigma = scm(sig)
        
        sig10 = generate_source(g0(XYZs1), Nsnaps, p1)
        sig20 = generate_source(g0(XYZs2), Nsnaps, p2) 
        noise0 = generate_noise(A0.shape[0], Nsnaps, sigma2)
        
        sig0 = sig10 + sig20 + noise0
        
            
        Sigma0 = scm(sig0)
    
        
        # init grid
        
        
        
        # box constraint for the positions
        
        # CMF and COMET2
        
        gasync = []
        Sigmaasync = []
        aa = []
        
        for m in range(Na):
            aa.append(np.copy(Arrayasync[Ns*m:Ns*m+Ns, :]))
        
            gasync.append(lambda x, a=aa[m] : freefield1D(a, x, Y, Z, k))
        
            Sigma_loc = scm(sig[Ns*m: Ns*m+Ns, m*Lseg:m*Lseg+Lseg])
        
            Sigmaasync.append(Sigma_loc)
            
        D_meas = block_diag(*Sigmaasync)
        
        
    

        Psiadmm = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()
        Psiagd = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()
        Psitnnr = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()

                    
        lamadmm = 1e-4   
        lamagd = 1e-4  
        lamtnnr = 1e-3
    
    
        Sagd = agd(D_meas, 1000, lamagd, Psiagd, Nsource)

    
        Nsnapsest = [Nsnaps//Na] * Na
        
    
        Xcmf, Pcmf, sigma2cmf = gridless_cmf(Sigmaasync, Nsnapsest,  gasync, Xinit, Nsource, box, mode="comet2")

        Xmle, Pmle, sigma2mle = mle_async(Sigmaasync, Nsnapsest, gasync, Xcmf, Pcmf, sigma2cmf, box)


        Xagd, Pagd, sigma2agd = gridless_cmf([Sagd], [Nsnaps], [g], Xinit, Nsource, box, mode="comet2")
        Xagdmle, Pagdmle, sigma2agdmle = mle_async([Sagd], [Nsnaps], [g], Xagd, Pagd, sigma2agd, box)

        
        XXcmf[:, npar] = Xcmf.ravel()
        XXagd[:, npar] = Xagd.ravel()

        XXmle[:, npar] = Xmle.ravel()
        XXagdmle[:, npar] = Xagdmle.ravel()
        
                
        PPcmf[:, npar] = Pcmf
        PPagd[:, npar] = Pagd

        PPmle[:, npar] = Pmle
        PPagdmle[:, npar] = Pagdmle

#%%
r = 40

plt.plot(K, np.ones(K.shape) * XYZs1[0], 'k', label='True X')
plt.plot(K, np.ones(K.shape) * XYZs2[0], 'k')


plt.scatter(K, XXcmf[0, :], marker='+', s=PPcmf[0, :] * r, facecolor="C1", label='A-COMET2')
plt.scatter(K, XXcmf[1, :], marker='+', s=PPcmf[1, :] * r, facecolor="C1")


plt.scatter(K, XXmle[0, :], marker='x', s=PPmle[0, :] * r, facecolor="C2", label='A-COMET2+ML')
plt.scatter(K, XXmle[1, :], marker='x', s=PPmle[1, :] * r, facecolor="C2")

plt.scatter(K, XXagdmle[0, :], marker='o', s=PPagdmle[0, :] * r, facecolor="none", edgecolor="C3", label='AGD+ML')
plt.scatter(K, XXagdmle[1, :], marker='o', s=PPagdmle[1, :] * r, facecolor="none", edgecolor="C3")


plt.ylim(-0.5, 1.)

plt.xlabel('Wavenumber $k$')
plt.ylabel("Position estimation X (m)")
plt.legend()
