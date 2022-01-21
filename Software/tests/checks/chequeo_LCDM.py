import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time

from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
##%matplotlib qt5

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)

sys.path.append('./Software/utils/')
from data import leer_data_pantheon

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb, zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

z = np.linspace(0,3,1000)
H_0 = 73.48
omega_m = 0.3

lala = H_LCDM(z,omega_m,H_0)
print(c_luz_km * lala**(-1))

#%%
M_abs = -18
muobs =  mb - M_abs
plt.figure()
plt.errorbar(zcmb,muobs, fmt='.k',label='observado')

for omega_m in (np.linspace(0,1,10)):
    H = H_LCDM(z,omega_m,H_0)
    #plt.plot(z,H_0 * E)
    #aux = np.zeros(len(E))
    d_c = cumtrapz(1/H,z,initial=0) #Paso c_luz a km/seg
    dc_int = interp1d(z,d_c) #Interpolamos
    d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    sn = len(muth)
    plt.plot(zcmb,muth,'.',label='omega={}'.format(omega_m))
    deltamu = muobs - muth
    plt.legend(loc='best')

    print(muth)

    #%%
    plt.figure()
    z = np.linspace(0,0.5,1000)
    for omega_m in (np.linspace(0,1,10)):
        H = H_LCDM(z,omega_m,H_0)
        plt.plot(z, H)
        plt.plot(z,H,label='omega={}'.format(omega_m))
        plt.legend(loc='best')
#%%
    plt.close()
    plt.figure()
    z = np.linspace(0,3,int(10**6))
    for omega_m in (np.linspace(0.2,0.4,3)):
        H = H_LCDM(z,omega_m,H_0=1)
        aux = cumtrapz(H**(-1),z,initial=0)
        plt.plot(z, aux,label='omega={}'.format(omega_m))
        #d_L = (1+z) * 0.001 * c_luz * cumtrapz(1/H,z,initial=0)
        #plt.plot(z, d_L)
        #plt.plot(z,H,label='omega={}'.format(omega_m))
        plt.legend(loc='best')

#%%
def integral(omega_m,z):
    zs = np.linspace(0,z,100000)
    H = H_LCDM(zs,omega_m,H_0=1)
    aux = cumtrapz(H**(-1),zs,initial=0)
    return aux[-1]

Z = 0.01
omega_m=0.2
H = H_LCDM(z,omega_m,H_0=1)
aux = cumtrapz(H**(-1),z,initial=0)
dc_int = interp1d(z,aux) #Interpolamos
#%%
print(dc_int(Z))
#%%
dc = np.zeros(len(zcmb))
for i,z in enumerate (zcmb):
    dc[i]=integral(0.2,z)
plt.plot(zcmb,dc_int(zcmb),'.')
plt.plot(zcmb,dc+0.001,'.')

#plt.figure()
#plt.plot(zcmb,deltamu,'.')
#plt.plot(zcmb,(muth-muobs)/muobs,'r.')
