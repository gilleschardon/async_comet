#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:50:31 2024

@author: chardon
"""
import numpy as np
import matplotlib.pyplot as plt

from utils import square_array




aperture = 0.5
N = 5

NPAR = [1,4,10]

for npar in NPAR:
    step = aperture / (N-1)
    delta = step*(npar)/2
    
    A1 = square_array(aperture, N, center=[delta/2, delta/2, 0], axis='z')
    A2 = square_array(aperture, N, center=[delta/2, -delta/2, 0], axis='z')
    A3 = square_array(aperture, N, center=[-delta/2, +delta/2, 0], axis='z')
    A4 = square_array(aperture, N, center=[-delta/2, -delta/2, 0], axis='z')
    
    epsilon = 0.0
    if npar == 4:
        epsilon = 0.015
    
    plt.figure()
    plt.scatter(A1[:, 0] + epsilon, A1[:, 1] + epsilon, marker='o')
    plt.scatter(A2[:, 0] - epsilon, A2[:, 1] + epsilon, marker='+')
    plt.scatter(A3[:, 0] + epsilon, A3[:, 1] - epsilon, marker='x')
    plt.scatter(A4[:, 0] - epsilon, A4[:, 1] - epsilon, marker='s')
    
    plt.axis("square")
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim((-0.65, 0.65))
    plt.ylim((-0.65, 0.65))
    
