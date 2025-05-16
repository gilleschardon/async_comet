#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:21:40 2020

@author: gilleschardon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import freefield
from utils import grid3D, scm
from scipy.linalg import block_diag, svd
from utils import gridless_cmf, mle_async
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from algosasync import bhmc, admm, apgl, agd, tnnradmm

def eval_perf(XYZs, XYZest, P, Pest):
    
    C = cdist(XYZs, XYZest)
    
    i1, i2 = linear_sum_assignment(C)

    errXYZ =np.sum(((XYZs[i1, :] - XYZest[i2, :]))**2)
    errP =np.sum((P[i1] - Pest[i2])**2)

    return errXYZ, errP


# Experimental data
mat = loadmat("data_sfw.mat")


# 4 quarters
renumbering = np.arange(128, dtype=np.int16)
Na = 4


#  8 bars
#renumbering = np.zeros([128], dtype=np.int16)

# for a in range(8):
#     renumbering[a*16:a*16+8] = np.arange(8) + a*8
#     renumbering[a*16+8:a*16+16] = np.arange(8) + a*8+64
# Na = 8



k = mat['k'][0,0] # wavenumber
Sigs = mat['data'][renumbering, :] # signals (a row of the STFT, for each microphone)
Sigs = Sigs / 1000 # to avoid numerical problems
Sigma = scm(Sigs) # Spatial covariance matrix
Array = mat['Pmic'][renumbering, :] # Array coordinates

#%%
# source modelmat[]
g = lambda x : freefield(Array, x, k)

# init grid
Xinit, dimgrid = grid3D([-2, -1, 4], [1, 0, 5], 0.05)

# box constraint for the positions
box = np.array([[-2, -1, 4],[1, 0, 5]])


Nsnaps = Sigs.shape[1]
Nsnapsest = [Nsnaps//Na] * Na
Nsource = 4
Ns = 128//Na
Lseg = Nsnaps//Na
gasync = []
Sigmaasync = []
aa = []

for m in range(Na):
    aa.append(np.copy(Array[Ns*m:Ns*m+Ns, :]))

    gasync.append(lambda x, a=aa[m] : freefield(a, x, k))

    Sigma_loc = Sigs[Ns*m: Ns*m+Ns, m*Lseg:m*Lseg+Lseg] @ Sigs[Ns*m: Ns*m+Ns, m*Lseg:m*Lseg+Lseg].T.conj()

    Sigmaasync.append(Sigma_loc)
    
D_meas = block_diag(*Sigmaasync)


P = np.ones([Nsource])

Xgt = np.array([[-1.6065,   -0.4699,    4.5994],[-0.8333  , -0.4910 ,   4.6172],[ 0.1476 ,  -0.4970  ,  4.5261],[-0.6867,   -0.7253   , 4.6501]])


lamadmm = 1e-4   
lamagd = 1e-4  
lamtnnr = 1e-3


Xpsi, dimpsi = grid3D([-2, -1, 4], [1, 0, 5], 0.1)
Aadmm = g(Xpsi)
ua, sa, va = svd(Aadmm)
Npsi = 40;
Psi = ua[:, :Npsi] @ ua[:, :Npsi].T.conj()


Sagd = agd(D_meas, 1000, lamagd, Psi, Nsource)



Xcmf, Pcmf, sigma2cmf = gridless_cmf(Sigmaasync, Nsnapsest,  gasync, Xinit, Nsource, box, mode="comet2")
Xmle, Pmle, sigma2mle = mle_async(Sigmaasync, Nsnapsest, gasync, Xcmf, Pcmf, sigma2cmf, box)

Xagd, Pagd, sigma2agd = gridless_cmf([Sagd], [Nsnaps], [g], Xinit, Nsource, box, mode="comet2")
  
Xagdmle, Pagdmle, sigma2agdmle = mle_async([Sagd], [Nsnaps], [g], Xagd, Pagd, sigma2agd, box)

methods = ['COMET2', 'COMET2+ML', 'TNNR', 'TNNR+ML', 'ADMM', 'ADMM+ML', 'AGD', 'AGD+ML']

Xests = (Xcmf, Xmle, Xagd, Xagdmle)

Nm = len(Xests)
ex = np.zeros([Nm])
for nm in range(Nm):
    
    Xest = Xests[nm]
    Pest = P
    
    
    ex[nm], ep = eval_perf(Xgt, Xests[nm], P, P)

    gest = g(Xest)
    
ind1 = np.argmin(ex[:2])
ind2 = np.argmin(ex[2:])+2

    
#%%
plt.figure()
for n in range(Na):
    
    plt.scatter(aa[n][:, 0], aa[n][:, 1])
    plt.axis('equal')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
    
plt.figure()
plt.subplot(2,1,1)
plt.scatter(Xgt[:, 0], Xgt[:, 1], label="Sources", marker='o')
plt.scatter(Xcmf[:, 0], Xcmf[:, 1], 100, label="COMET2", marker='x')
plt.scatter(Xmle[:, 0], Xmle[:, 1], 100, label="COMET2+ML", marker='+')

plt.scatter(Xests[ind2][:, 0], Xests[ind2][:, 1], label=methods[ind2], marker='s')

plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.axis('equal')
plt.legend()


plt.subplot(2,1,2)
plt.scatter(Xgt[:, 0], Xgt[:, 2], label="Sources", marker='o')
plt.scatter(Xcmf[:, 0], Xcmf[:, 2], 100, label="COMET2", marker='x')
plt.scatter(Xmle[:, 0], Xmle[:, 2], 100, label="COMET2+ML", marker='+')


plt.scatter(Xests[ind2][:, 0], Xests[ind2][:, 2], label=methods[ind2], marker='s')

plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.axis('equal')



