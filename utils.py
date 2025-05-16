#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:29:40 2025

@author: chardon
"""

import numpy as np
from scipy.linalg import inv
import scipy.optimize
from numpy.random import normal
from scipy.linalg import sqrtm, eigh, pinv



def square_array(aperture, N, center = [0,0,0], axis='z'):
    
    c = np.linspace(-aperture/2, aperture/2, N)
    
    if axis == 'x':
        xg, yg, zg = np.meshgrid(center[0], c + center[1], c + center[2])
    elif axis == 'y':
        xg, yg, zg = np.meshgrid(c + center[0], center[1], c + center[2])
    elif axis == 'z':
        xg, yg, zg = np.meshgrid(c + center[0], c + center[1], center[2])     
        
    return np.vstack([xg.ravel(), yg.ravel(), zg.ravel()]).T

def freefield(PX, PS, k, normalized=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        dz = PX[:, 2] - PS[2]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx*dx + dy*dy + dz*dz);
        
    D = np.exp( -1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D

def freefield_derivatives(PX, PS, k):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        dy = PX[: ,1] - PS[1]
        dz = PX[:, 2] - PS[2]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T
        dy = PX[:, 1:2] - PS[:, 1:2].T
        dz = PX[:, 2:3] - PS[:, 2:3].T

    d = np.sqrt(dx*dx + dy*dy + dz*dz);
        
    D = np.exp( -1j * k * d) / d

    Dd = -1j * k * D - D/d
    
    Dx = - Dd * dx / d
    Dy = - Dd * dy / d
    Dz = - Dd * dz / d
        
    return D, Dx, Dy, Dz

def freefield1D(PX, PS, Y, Z, k, normalized=False):
      
    if PS.ndim == 1:
        dx = PX[:, 0] - PS[0]
        
    else:    
        dx = PX[:, 0:1] - PS[:, 0:1].T

    d = np.sqrt(dx*dx + Y*Y + Z*Z);
        
    D = np.exp( -1j * k * d) / d
    
    if normalized:
        D = D / np.sqrt(np.real(np.sum(D * np.conj(D), axis=0)))
        
    return D

def gridless_cmf(Sigmas, Nsnaps, gs, X_init, Niter, box, A_inits=None, mode="comet2"):


    # generates the dictionary on the grid, if not given as an argument
    if not A_inits:
        A_inits = [g(X_init) for g in gs]
        # A_inits = []
        # for ns in range(Nsync):
        #     A_inits.append(gs[ns](X_init))
        
        
    if mode == 'comet1':
        Sigmas_inv = [inv(Sigma) for Sigma in Sigmas]

        # Sigmas_inv = []
        # for ns in range(Nsync):
        #     Sigmas_inv.append(inv(Sigmas[ns]))
    else:
        Sigmas_inv = [None for Sigma in Sigmas]
        
    #A_inits_norm = []    
    
    A_inits_norm = [A_init / np.sqrt(np.real(np.sum(A_init * np.conj(A_init), axis=0))) for A_init in A_inits]


    # for ns in range(Nsync):
    #     A_init = A_inits[ns]
    #     A_inits_norm.append(A_init / np.sqrt(np.real(np.sum(A_init * np.conj(A_init), axis=0))))
            
    # spatial dimension
    dim = X_init.shape[1]

    # variables
    Xs = np.zeros([0, dim])
    Ps = np.zeros([0])
    sigma2 = 1
    lsigma2 = 0.001 # lower bound on sigma2, to avoid singular covariance matrices

    for iter in range(Niter):
        
        #print(f"Iter {iter+1}/{Niter}")
        
        # select a new source
        Xnew, nu = maximize_nu_cmf(Sigmas, Nsnaps, Sigmas_inv, gs, Xs, Ps, sigma2, lsigma2, X_init, A_inits_norm, box, mode)

        Xs = np.vstack([Xs, Xnew])

                
        # optimize the amplitudes
        Ps, sigma2 = optimize_amplitudes_cmf(Sigmas, Nsnaps, Sigmas_inv, gs, Xs, Ps, sigma2, lsigma2, mode)


        # optimize the amplitudes and positions
        Xs, Ps, sigma2 = optimize_amplitudes_positions_cmf(Sigmas, Nsnaps, Sigmas_inv, gs, Xs, Ps, sigma2, lsigma2, box, dim, mode)
        
    return Xs, Ps, sigma2

def maximize_nu_cmf(Sigmas, Nsnaps, Sigmas_inv, gs, Xs, Ps, sigma2, lsigma2, X_init, A_inits, box, mode):
    # maximizes the nu criterion for a new source
    
    
    Glocs = [g(Xs) for g in gs]
    #Gloc = g(Xs)
    sigma2p = lsigma2+sigma2
    
    if mode == 'cmf':
        Sigmas_est = [Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Gloc.shape[0]) for Gloc in Glocs]
        #Sigma_est = Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Sigma.shape[0])
        RR = []
        for ns in range(len(Sigmas)):
            RR.append(Sigmas[ns] - Sigmas_est[ns])
    else:
        if Glocs[0].shape[1] > 0:    
            Sigma_est_invs = [np.eye(Gloc.shape[0]) / sigma2p - Gloc @ inv(np.diag(1/Ps) + Gloc.T.conj() @ Gloc / (sigma2p)) @ Gloc.T.conj() / sigma2p**2 for Gloc in Glocs]
                
        else:
            Sigma_est_invs = [np.eye(Gloc.shape[0]) / sigma2p for Gloc in Glocs]# no source

        if mode == 'comet1':
            #Sigma_ests = [Gloc @ (Gloc * Ps).T.conj() + sigma2p * np.eye(Gloc.shape[0]) for Gloc in Glocs]

            RR = []
            for ns in range(len(Sigmas)):
                RR.append(Sigma_est_invs[ns] @ Sigmas[ns] @ Sigma_est_invs[ns] - Sigmas_inv[ns])
        else:
            RR = []
            for ns in range(len(Sigmas)):
                RR.append(Sigma_est_invs[ns] @ Sigmas[ns] @ Sigmas[ns] @ Sigma_est_invs[ns] - np.eye(Glocs[ns].shape[0]))
    
    # global optimization on a grid
    nugrid = np.zeros([X_init.shape[0]])
                       
    for ns in range(len(Nsnaps)):             
        nugrid = nugrid + Nsnaps[ns] * np.real(np.sum( A_inits[ns].conj() * (RR[ns] @ A_inits[ns]), axis=0))
    idx = np.argmax(nugrid)
    Xgrid = X_init[idx, :]
    
    # local optimization
    
    # objective function
    def nuobj(X):  
        nu = 0
        for ns in range(len(Nsnaps)):
            gx = gs[ns](X)    
            gx = gx / np.sqrt(np.sum(gx*gx.conj()))
            
            nu = nu + Nsnaps[ns] * (- np.real(gx.T.conj() @ RR[ns] @ gx))
        return nu
   
    boxbounds = scipy. optimize.Bounds(box[0, :]+0.01, box[1, :]-0.01)    
    options = {"disp": False, "ftol": 1e-12, "gtol": 1e-12}

    res = scipy.optimize.minimize(nuobj, Xgrid, bounds=boxbounds, options=options)

    return res.x, - res.fun

def optimize_amplitudes_cmf(Sigmas, Nsnaps, Sigma_invs, gs, Xs, Ps, sigma2, sigma2l, mode):    
    Glocs = [g(Xs) for g in gs]

    objfunamp = lambda x : amplitudesobj_cmf(Sigmas, Nsnaps, Sigma_invs, Glocs, x, sigma2l, mode)
    init = np.hstack([Ps, 0.001, sigma2])   
    options = {"disp": False, "ftol": 1e-12, "gtol": 1e-12}

    res = scipy.optimize.minimize(objfunamp, init, bounds=scipy.optimize.Bounds(np.zeros([Ps.shape[0]+2])), options=options)
    return res.x[:-1], res.x[-1]
    
def amplitudesobj_cmf(Sigmas, Nsnaps, Sigmas_inv, Glocs, x, sigma2l, mode):
    


    # unpacking    
    Ps = x[:-1]
    sigma2p = x[-1]
    
    Sigmas_est = [Gloc @ (Gloc * Ps).T.conj() + (sigma2l+sigma2p) * np.eye(Gloc.shape[0]) for Gloc in Glocs]

    obj = 0
    
    if mode=='cmf':
        for ns in range(len(Nsnaps)):
            obj = obj + Nsnaps[ns] * np.sum(np.abs(Sigmas[ns]-Sigmas_est[ns])**2)
        return obj
    
    # avoids instability with almost zero powers
    sup = Ps > 0.0001
    
    if np.max(Ps) > 0.0001:
        Sigma_est_invs = [np.eye(Gloc.shape[0]) /  (sigma2l+sigma2p) - Gloc[:, sup] @ inv(np.diag(1/Ps[sup]) + Gloc[:, sup].T.conj() @ Gloc[:, sup] / ( (sigma2l+sigma2p))) @ Gloc[:, sup].T.conj() /  (sigma2l+sigma2p)**2  for Gloc in Glocs]

    else:
        Sigma_est_invs = [np.eye(Gloc.shape[0]) /  (sigma2l+sigma2p) for Gloc in Glocs]
        
    obj = 0    
        
    if mode == 'comet1':
        for ns in range(len(Nsnaps)):
            obj = obj + Nsnaps[ns] * np.real( np.trace(Sigma_est_invs[ns] @ Sigmas[ns] + Sigmas_est[ns] @ Sigmas_inv[ns]))
    else:
        for ns in range(len(Nsnaps)):
               obj = obj + Nsnaps[ns] *    np.real(np.trace(Sigmas[ns] @ Sigma_est_invs[ns] @ Sigmas[ns]) + np.trace(Sigmas_est[ns]))        

    return obj

def optimize_amplitudes_positions_cmf(Sigmas, Nsnaps, Sigma_invs, gs, Xs, Ps, sigma2, sigma2l, box, dim, mode):
    
    xinit = np.hstack([Xs.ravel(), Ps, sigma2])
    Ns = Ps.shape[0]
    
    lb = np.hstack([ np.kron(np.ones([Ns]), box[0, :]), np.zeros([Ns+1])])
    ub = np.hstack([ np.kron(np.ones([Ns]), box[1, :]), np.inf * np.ones([Ns+1])])
    boxbounds = scipy.optimize.Bounds(lb, ub)

    objfun = lambda x : ampposobj_cmf(Sigmas, Nsnaps, Sigma_invs, gs, x, sigma2l, dim, mode)
    options = {"disp": False, "ftol": 1e-12, "gtol": 1e-12}
    res = scipy.optimize.minimize(objfun, xinit, bounds=boxbounds, options=options)
   
    # unpacking
    Xs = np.reshape(res.x[:dim*Ns], [Ns, dim])
    Ps = res.x[dim*Ns : -1]
    sigma2 = res.x[-1]

    return Xs, Ps, sigma2

def ampposobj_cmf(Sigmas, Nsnaps, Sigma_invs, gs, x, sigma2l, dim, mode):
    
    Ns = (x.shape[0] - 1)// (dim + 1)

    # unpacking
    X = np.reshape(x[:dim*Ns], [Ns, dim])
    Psigma2 = x[dim*Ns :]
  
    obj = 0
    Glocs = [g(X) for g in gs]

    obj = amplitudesobj_cmf(Sigmas, Nsnaps, Sigma_invs, Glocs, Psigma2, sigma2l, mode)
    
    return obj

def mle_async(Sigmas, Nsnaps, gs, Xs, Ps, sigma2, box):
    
    sigma2l = 0.0001
    dim = box.shape[1]
    
    xinit = np.hstack([Xs.ravel(), Ps, np.maximum(0,sigma2 - sigma2l)])
    Ns = Ps.shape[0]
    
    lb = np.hstack([ np.kron(np.ones([Ns]), box[0, :]), np.zeros([Ns+1])])
    ub = np.hstack([ np.kron(np.ones([Ns]), box[1, :]), np.inf * np.ones([Ns+1])])
    boxbounds = scipy.optimize.Bounds(lb, ub)

    objfun = lambda x : mlefunc(Sigmas, Nsnaps, gs, x, sigma2l, dim)
    options = {"disp": False, "ftol": 1e-12, "gtol": 1e-12}
    res = scipy.optimize.minimize(objfun, xinit, bounds=boxbounds, options=options)
   
    # unpacking
    Xs = np.reshape(res.x[:dim*Ns], [Ns, dim])
    Ps = res.x[dim*Ns : -1]
    sigma2 = res.x[-1]

    return Xs, Ps, sigma2

def mlefunc(Sigmas, Nsnaps, gs, x, sigma2l, dim):
    
    Ns = (x.shape[0] - 1)// (dim + 1)

    # unpacking
    X = np.reshape(x[:dim*Ns], [Ns, dim])
    Psigma2 = x[dim*Ns :]
  
    Ps = Psigma2[:-1]
    sigma2p = Psigma2[-1]
    
    
    obj = 0
    Glocs = [g(X) for g in gs]

    Sigmas_est = [Gloc @ (Gloc * Ps).T.conj() + (sigma2l+sigma2p) * np.eye(Gloc.shape[0]) for Gloc in Glocs]
    
    
    sup = Ps > 0.0001
    
    if np.max(Ps) > 0.0001:
        Sigma_est_invs = [np.eye(Gloc.shape[0]) /  (sigma2l+sigma2p) - Gloc[:, sup] @ inv(np.diag(1/Ps[sup]) + Gloc[:, sup].T.conj() @ Gloc[:, sup] / ( (sigma2l+sigma2p))) @ Gloc[:, sup].T.conj() /  (sigma2l+sigma2p)**2  for Gloc in Glocs]

    else:
        Sigma_est_invs = [np.eye(Gloc.shape[0]) /  (sigma2l+sigma2p) for Gloc in Glocs]
        

    for ns in range(len(Nsnaps)):
        _, ld = np.linalg.slogdet(Sigmas_est[ns])
        obj = obj + Nsnaps[ns] * ( np.trace( Sigma_est_invs[ns] @ Sigmas[ns]) + ld)

    obj = np.real(obj)
    return obj

def grid3D(lb, ub, step):
    xx = np.linspace(lb[0], ub[0], int((ub[0]-lb[0]) // step + 1))
    yy = np.linspace(lb[1], ub[1], int((ub[1]-lb[1]) // step + 1))
    zz = np.linspace(lb[2], ub[2], int((ub[2]-lb[2]) // step + 1))

    dims = [xx.size, yy.size, zz.size]

    Xg, Yg, Zg = np.meshgrid(xx, yy, zz)
    
    return np.vstack([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T, dims

def generate_noise(Nm, Ns, sigma2):
    noise = normal(scale = np.sqrt(sigma2/2), size=(Nm, Ns)) + 1j * normal(scale = np.sqrt(sigma2/2), size=(Nm, Ns))    
    return noise

def generate_source(g, Ns, p):
    sig = g[:, np.newaxis] @ (normal(scale = np.sqrt(p/2), size=(1, Ns)) + 1j * normal(scale = np.sqrt(p/2), size=(1, Ns)))    
    return sig

def scm(sig):
    return sig @ sig.conj().T / sig.shape[1]



def CRB_unc_Nsources_async(Nsnaps, Xs, p, sigma2, sourcemodels):
    
    Ns = Xs.shape[0]
    Na = len (sourcemodels)    
    
    
    F = np.zeros((4 * Ns + 1, 4 * Ns + 1, Na))
    
    for na in range(Na):
        
        a, dx, dy, dz = sourcemodels[na](Xs)
        
        Nm = a.shape[0]
        
        
        Sigma = a @ np.diag(p) @ a.T.conj() + sigma2 * np.eye(Nm)
        Sigmainv = inv(Sigma)
        
        
        Sigmadiff = np.zeros((Nm, Nm, 4, Ns), dtype=np.complex128)
        
        for ns in range(Ns):
        
            Sigmadiff[:, :, 0, ns] = a[:, ns:ns+1] @ a[:, ns:ns+1].T.conj()
            Sigmadiff[:, :, 1, ns] = p[ns] * (a[:, ns:ns+1] @ dx[:, ns:ns+1].T.conj() + dx[:, ns:ns+1] @ a[:, ns:ns+1].T.conj())
            Sigmadiff[:, :, 2, ns] = p[ns] * (a[:, ns:ns+1] @ dy[:, ns:ns+1].T.conj() + dy[:, ns:ns+1] @ a[:, ns:ns+1].T.conj())
            Sigmadiff[:, :, 3, ns] = p[ns] * (a[:, ns:ns+1] @ dz[:, ns:ns+1].T.conj() + dz[:, ns:ns+1] @ a[:, ns:ns+1].T.conj())
            
        for ns in range(Ns):
            for ms in range(Ns):
                for u in range(4):
                    for v in range(4):
                        F[u + ns * 4, v + ms * 4, na] = Nsnaps[na] * np.real(np.trace(Sigmainv @ Sigmadiff[:, :, u, ns] @ Sigmainv @ Sigmadiff[:, :, v, ms]))
                        
        for ns in range(Ns):
            for u in range(4):
                F[u + ns * 4, -1, na] = Nsnaps[na] * np.real(np.trace(Sigmainv @ Sigmadiff[:, :, u, ns] @ Sigmainv))
                F[-1, u + ns * 4, na] = F[u + ns * 4, -1, na]
                
        F[-1, -1, na] = Nsnaps[na] * np.real(np.trace(Sigmainv @ Sigmainv))

    Fglob = np.sum(F, axis=2)
    
    return inv(Fglob)
