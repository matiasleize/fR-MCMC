"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador

#%%
omega_m = 0.24
b = 1
H0 = 73.48
params_fisicos = [omega_m,b,H0]

cantidad_zs = int(10**6)

max_steps = np.linspace(0.001,1,5)#1000
Hs = []
for maxs in max_steps:
    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cantidad_zs,
    max_step=maxs)
    Hs.append(H_ode)

final = []
for j in range(1,len(Hs)):
    aux = np.mean((Hs[j]-Hs[j-1])/Hs[j]);
    final.append(aux);
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('max_steps')
plt.ylabel('%Hs')
plt.plot(max_steps[1:],np.array(final)*100,'.');
#plt.legend(loc='best')
plt.show()

final
max_steps
