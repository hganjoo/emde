import streamlit

# EMDE Transfer Functions

import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.special import hyp2f1,gamma,digamma
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sys as ss

streamlit.header('Early Matter-Dominated Era Transfer Functions')

streamlit.write('This page provides transfer functions for the matter power spectrum after a period of early matter domination (EMD) caused by a long-lived hidden sector particle (Y) that is initially relativistic.\nMultiply this transfer function with your favorite base power spectrum to generate a power spectrum that includes the effects of the EMD and the small-scale cut-off due to the relativistic pressure of the Y particles in the early Universe.')



streamlit.sidebar.title('Choose your parameters:\n')

streamlit.sidebar.markdown('EMDE parameters:\n')
streamlit.sidebar.markdown('$T_{RH}$ in GeV')
Trh = streamlit.sidebar.number_input('',min_value=0.01)
streamlit.sidebar.markdown(r'$$\eta$$: The ratio of SM to Y particle density at initial time')
eta = streamlit.sidebar.number_input('',min_value=1e-6,value=5.0,step=1e-6)
streamlit.sidebar.markdown('$$k_{MD} / k_{RH}$$: This is $k_{dom} / k_{RH}$ if $\eta>1$, else $k_y/k_{RH}$')
k_emde = streamlit.sidebar.number_input('',min_value=5,value=100)
streamlit.sidebar.markdown('Y Particle Statistics')
yp_st = streamlit.sidebar.selectbox('',['Boson','Fermion'])

streamlit.sidebar.markdown('Basic cosmology:\n')
streamlit.sidebar.markdown(r'$$\Omega_m$$')
OmegaM = streamlit.sidebar.number_input('',min_value=0.2,value=0.3089)
streamlit.sidebar.markdown(r'$$\Omega_b$$')
OmegaB = streamlit.sidebar.number_input('',min_value=0.01,value=0.04)
streamlit.sidebar.markdown(r'$$h$$')
h = streamlit.sidebar.number_input('',min_value=0.5,max_value=0.8,value=0.67)



if yp_st == 'Boson': 
    st=-1
else:
    st=1


pi = np.pi

def arh_r(T): return (1/1.02) * np.power(gstar(T0)/gstar(0.34*T),1./3) * (T0/T)

def H_RD(T): return np.sqrt((8*pi*pi*pi/90)*gstar(T))*T*T/mpl


mpl = 1.221e19 #GeV
T0 = 2.348e-13 #GeV

G = 1./(mpl*mpl)

sec_inv_GeV = 6.58e-25
GeV_in_Mpc_inv = (1.97e-16 * 3.24078e-23)**-1.0

rho_conv = 34.238e56 # Convert Gev^4 to Msun / (Mpc)^3

gstar_file = np.loadtxt('gstar.dat')
temps = gstar_file[:,0] # Temps loaded in GeV
gstars = gstar_file[:,1]

gstar = interp1d(temps,gstars,fill_value=(gstars[0],gstars[-1]),bounds_error=False)


OmegaC = OmegaM - OmegaB


if eta > 1: kd_krh = k_emde
if eta < 1: ky_krh = k_emde
#st = 1

g = 1.
b = 2.7
p = 1.


if st == 1:

    b = 3.15
    p = 7./8
    



# AE+ 2011 TFs
# Model transition from log to EMDE in k-space

# y=k/k_rh, xdec = k_rh/keq, x = k/keq = y*xdec, y = x/xdec

k = np.logspace(-3,12,1000) # 1 / Mpc


keq = 0.073 * OmegaM * h * h # 1 / Mpc

krh = arh_r(Trh) * H_RD(Trh) * GeV_in_Mpc_inv # 1/Mpc

xdec = krh/keq



Step = lambda y: .5*(np.tanh(.5*y)+1)
Afit = lambda y: np.exp(0.60907/(1 + 2.149951*(-1.51891 + np.log(y))**2)**1.3764)*(9.11*Step(5.02 - y) + 3./5*y**2*Step(y - 5.02))
Bfit = lambda y: np.exp(np.log(0.594)*Step(5.02 - y) + np.log(np.e/y**2)*Step(y - 5.02))
aeqOVERahor = lambda x,y: x*np.sqrt(2)*(1 + y**4.235)**(1/4.235)
  

#fb = OmegaB/OmegaM
#f1 = 1 - 0.568*fb + 0.094*fb**2
#f2 = 1 - 1.156*fb + 0.149*fb**2 - 0.074*fb**3
#f2_over_f1 = f2/f1

a1 = (1-(1+24*OmegaC/OmegaM)**.5)/4.
a2 = (1+(1+24*OmegaC/OmegaM)**.5)/4.
B = 2*digamma(1)-digamma(a2)-digamma(a2+.5)
f2_over_f1 = B/np.log(4/np.e**3)

def Rt(xdec,x):
    return (Afit(x/(xdec))*np.log((4/np.exp(3))**f2_over_f1*Bfit(x/(xdec))*aeqOVERahor(x, x/(xdec))))/(9.11*np.log((4/np.exp(3))**f2_over_f1*0.594*x*np.sqrt(2)))
  

if eta > 1:
    
    from smd_cutfuncs import ky,kc,tk_ss
    
    kcut = kc(kd_krh,eta,b)
    k = np.logspace(-3,np.log10(1000*kcut*krh),10000)
    x = k/keq
    y = k/krh
    tk = np.where(x<0.05*xdec,1,Rt(xdec,x)) 
    q = (k/krh)/(kd_krh)
    tk = tk * np.log(1 + 0.22*q) * np.power(1 + 1.11*q + (0.94*q)**2 + (0.63*q)**3 + (0.45*q)**4,-0.25) / (0.22*q)
    tk = tk * tk_ss(y,kd_krh,eta,b)
    
if eta < 1:
    
    from yd_cutfuncs import npred_yd,kc,tk_ss
    
    kcut = kc(ky_krh,eta,b)
    k = np.logspace(-3,np.log10(1000*kcut*krh),10000)
    x = k/keq
    y = k/krh
    tk = np.where(x<0.05*xdec,1,Rt(xdec,x)) 
    tk = tk * tk_ss(y,ky_krh,eta,b)
    
  

    

fig,ax = plt.subplots(dpi=300) 
ax.loglog(k,tk,lw=3)
plt.ylim(1e-2,2*tk.max()) 
plt.xlim(k[0],2*k[np.where(tk>1e-2)[0][-1]])
plt.xlabel(r'k [Mpc$^{-1}$]')
plt.ylabel(r'$R_{\rm EMD}(k)$')
plt.title('EMD Transfer Function')

df = pd.DataFrame([k,tk]).transpose()
df.columns = ['k','R(k)']

streamlit.download_button('Download as CSV',data=df.to_csv(index=False),file_name='emde_tk.csv')

streamlit.pyplot(fig)

streamlit.write('EMDE transfer functions taken from Erickcek and Sigurdson 2011 (https://arxiv.org/abs/1106.0536).')

streamlit.write('Basic code implementing 2011 transfer functions by Adrienne Erickcek and M. Sten Delos. Modified by Himanish Ganjoo.')

streamlit.write('Modifications for initial radiation domination and cut-off scale from Ganjoo et al (). Check paper for parameter definitions.')


   
 


