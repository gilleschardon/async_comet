#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:03:20 2024

@author: chardon
"""

import numpy as np

from scipy.linalg import eigh, block_diag, svd
import matplotlib.pyplot as plt
def bhmc(Sigmas):
    
    vvs = []
    
    for n in range(len(Sigmas)):
       l, v = eigh(Sigmas[n])
       
       vvs.append(v @ np.diag(np.sqrt(l)))
       
    U = block_diag(*vvs)
    
    M = np.tile(np.eye(Sigmas[0].shape[0]), [len(Sigmas), len(Sigmas)])
    
    return U @ M @ U.T.conj()
#%%
def admm(Sm, Psi, mIter, gamma, mu, lamb):
    
    Omega = (Sm == 0)
    Inv_Omega = np.logical_not(Omega)
    
    N = Omega.shape[0]
    
    M = np.random.randn(N, N)/100
    
    Y = np.zeros([N,N])

    for iter in range(mIter):
        G = M - (1/mu) * Y

        
        l, v = eigh(G)
        t = lamb / mu
        ll = np.maximum(l - t, 0)
        S = v @ np.diag(ll) @ v.T.conj()
        S = Psi @ S @ Psi
        E = S + Y/mu
        
        Z = np.zeros([N,N])
        Z = Z + ((Sm + mu * E) / (mu + 1) ) * Inv_Omega
        Z = Z + E * Omega
        
        M = Z
        
        Y = Y + lamb * mu * (S - M)
       
    
    return S

def tnnradmm(Sm, Psi, mIter, mu, Ns):
    l, v = eigh(Sm)
    A = v[:, -Ns:]
    
    
    Omega = (np.abs(Sm) > 0)
    Inv_Omega = np.logical_not(Omega)
    
    N = Omega.shape[0]
    
    Z = np.random.randn(N, N)/100
    
    Y = np.zeros([N,N])
    
    

    for iter in range(mIter):
        G = Z - (1/mu) * Y

        
        U, s, Vh = svd(G)
        t = 1 / mu
        ll = np.maximum(s - t, 0)
        S = U @ np.diag(ll) @ Vh
        S = Psi @ S @ Psi.T.conj()
        
        Z = S + (A@A.T.conj() + Y) / mu
        
        
        Z = Omega * Sm + Inv_Omega * Z
        
        
        Y = Y + mu * (S - Z)  
        
    
    return S

def apgl(Sm, niter, lamb, Psi, Ns):
    
    Omega = np.logical_not(Sm == 0)

    l, v = eigh(Sm)
    A = v[:, -Ns:]
    
    Sc = Sm
    Scm = Sc
    Y = Sc
    t = 1

    for k in range(niter):
        G = Y + t * (A @ A.T.conj() - lamb * (Omega * Y - Omega*Sm))
        l, v = eigh(G)
        ll = np.maximum(l - t, 0)
        Scp = v @ np.diag(ll) @ v.T.conj()
        Scp = Psi @ Scp @ Psi
        tp = (1 + np.sqrt(1 + 4*t**2))/2
        Y = Scp + (t-1)/tp * (Sc - Scm)
        Scm = Sc
        Sc = Scp
        t = tp
        
    return Scp

def agd(Sm, niter, lamb, Psi, Ns):
    
    beta = 0.99
    l, v = eigh(Sm)
    
    L = v @ np.diag(np.sqrt(l))
    
    alpha = 0.5/np.max(l)
    t = 1
    Omega = np.logical_not(Sm == 0)

    for iter in range(niter):
        
        L = Psi @ L
        Y = (1 - alpha*lamb) * L - alpha * (Omega*(L@L.T.conj()) - Sm)@L
        tp = (np.sqrt(t**4 + 4 * t**2) - t**2)/2
        L = Y + tp*(1-t)/t * (Y - L)
        
        C = L @ L.T.conj()
        
        t = tp
        
        alpha = beta * alpha
        
        
    C = (C + C.T.conj())/2    
    return C