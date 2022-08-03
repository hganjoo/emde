#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:54:28 2022

@author: hganjoo
"""

import numpy as np
from scipy.special import lambertw as W

n = 2.7

def ky(kd,xi,b):
    
    return b*xi*kd*np.sqrt(1 + 1./xi)/1.414

def kpk(kd,xi,b):
    
    if b == 2.7: a = 1.26
    if b == 3.15: a = 1.28
    
    if xi > 50:
        
        
        X = np.power(3*0.594*xi/1.43,2./3)
        return ky(kd,xi,b) * (a/b)*np.real(np.power(W(0.67*X),-1.5))
    
    else:
        
        return ky(kd,xi,b) * ((a+0.58)/b)*np.power(xi,-0.299)
    

def kc(kd,xi,b):
    
    if b == 2.7: a = 1.26
    if b == 3.15: a = 1.28
    
    if xi > 50:
    
        X = np.power(np.real(W(0.77*xi**(2./3))),-1.5)
        x = ((0.58+a)/b) * X * np.power(np.log(0.18*xi*X),1/n) * np.sqrt(1 + 1/xi)
        
        return ky(kd,xi,b) * x
    
    else:
        
        return ky(kd,xi,b) * ((a + 0.71)/b)*np.power(xi,-0.21)
    
def tk_ss(k,kd,xi,b):
    
    kcut = kc(kd,xi,b)
    ks = k/kcut
    
    return np.exp(-1*np.power(ks,n))

