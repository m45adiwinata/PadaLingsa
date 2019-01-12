# -*- coding: utf-8 -*-
"""
Created on Tue May 22 23:27:11 2018

@author: Grenceng
"""
import numpy as np
import math
import dft as DFT

def stftAnal(x, fs, w, N, H):
    M = w.size
    hM1 = int(math.floor((M+1)/2))
    hM2 = int(math.floor(M/2))
    x = np.append(np.zeros(hM2),x)
    x = np.append(x,np.zeros(hM2))
    pin = hM1
    pend = x.size-hM1
    w = w / sum(w)
    y = np.zeros(x.size)
    while pin<=pend:
        x1 = x[pin-hM1:pin+hM2]                     # framing/pemotongan bagian x
        mX, pX = DFT.dftAnal(x1, w, N)              # analisis DFT pada frame
        if pin == hM1:
            xmX = np.array([mX])
            xpX = np.array([pX])
        else:
            xmX = np.vstack((xmX,np.array([mX])))
            xpX = np.vstack((xpX,np.array([pX])))
        pin += H
    return xmX, xpX
