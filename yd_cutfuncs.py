

import numpy as np

def kpk(ky,xi,b):
    
    return ky * (2.06/b) / np.sqrt(1 + xi)

def npred_yd(xi):
    if xi <= 0.1:
        return 2.2
    else:
        return 2.2 - 0.29*(xi - 0.1)
    

def kc(ky,xi,b):
    
    nn = npred_yd(xi)
    return kpk(ky,xi,b)*np.power(0.5*nn,1./nn)
    
def tk_ss(k,ky,xi,b):
    
    kcut = kc(ky,xi,b)
    ks = k/kcut
    
    return np.exp(-1*np.power(ks,npred_yd(xi)))