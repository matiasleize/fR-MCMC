import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
from scipy.integrate import cumtrapz as cumtrapz
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

#%%
omega_m = 0.24
H0 = 73.48

#bs = np.linspace(0.1, 5, 3)
bs = [3, 6]
cantidad_zs = int(10**3)
max_steps = 10**(-4)

Hs = np.zeros((len(bs),cantidad_zs))

for i, b in enumerate(bs):
    params_fisicos = [omega_m,b,H0]
    zs, H_ode = integrador(params_fisicos, n=2, cantidad_zs=cantidad_zs,
                max_step=max_steps)
    Hs[i,:] = H_ode

#%%
H_lcdm = H_LCDM(zs, omega_m, H0)
Da_lcdm = (c_luz_km/(1 +zs)) * cumtrapz(H_lcdm**(-1), zs, initial=0)
Dl_lcdm = (c_luz_km*(1 +zs)) * cumtrapz(H_lcdm**(-1), zs, initial=0)

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)

#plt.title('Modelo de Hu-Sawicki con $n=1$', fontsize=15)
plt.title('Modelo de Starobinsky con $n=1$', fontsize=15)

plt.xlabel('$z$ (Redshift)', fontsize=13)
plt.ylabel('$H(z)$ (Parámetro de Hubble)', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(zs[zs<3], H_lcdm[zs<3], label='b = 0')

for i in range(len(bs)):
    plt.plot(zs[zs<3],Hs[i,:][zs<3],label='b = {}'.format(bs[i]));

plt.legend(loc='best',prop={'size': 12})
plt.show()


#%% Distancia Luminosa (DL)
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)

#plt.title('Modelo de Hu-Sawicki con $n=1$', fontsize=15)
plt.title('Modelo de Starobinsky con $n=1$', fontsize=15)

plt.xlabel('$z$ (Redshift)', fontsize=13)
plt.ylabel('$d_L(z)$ (Distancia luminosa)', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(zs[zs<3], Dl_lcdm[zs<3], label='b = 0')

for i in range(len(bs)):
    Dl = (c_luz_km*(1 +zs)) * cumtrapz(Hs[i,:]**(-1), zs, initial=0)
    plt.plot(zs[zs<3],Dl[zs<3],label='b = {}'.format(bs[i]));

plt.legend(loc='best',prop={'size': 12})
plt.show()


#%% Distancia luminosa angular (DA)
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)

#plt.title('Modelo de Hu-Sawicki con $n=1$', fontsize=15)
plt.title('Modelo de Starobinsky con $n=1$', fontsize=15)

plt.xlabel('$z$ (Redshift)', fontsize=13)
plt.ylabel('$D_{A}(z)$ (Distancia luminosa angular)', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(zs[zs<3], Da_lcdm[zs<3], label='b = 0')

for i in range(len(bs)):
    Da = (c_luz_km/(1 +zs)) * cumtrapz(Hs[i,:]**(-1), zs, initial=0)
    plt.plot(zs[zs<3],Da[zs<3],label='b = {}'.format(bs[i]));

plt.legend(loc='best',prop={'size': 12})
plt.show()
