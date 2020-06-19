"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from scipy.interpolate import interp1d


import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon
from funciones_LambdaCDM import params_to_chi2
#%%

min_z = 0
max_z = 3

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt'
                        ,min_z=min_z,max_z=max_z)
sn = len(zcmb)
Cinv.shape
#%%
#Parametros a ajustar
#M_true = -19.5
#omega_m_true = 0.3089
H0_true =  73.48 #Unidades de (km/seg)/Mpc
#H0_true =  69.01 #Unidades de (km/seg)/Mpc
omegas = np.linspace(0,1,140)
ms = np.linspace(-20.3,-18.3,40)

chies = np.zeros((len(omegas),len(ms)))

for i,omega_m in enumerate(omegas):
    for j,M_abs in enumerate(ms):
        theta = [M_abs,omega_m]
        chies[i,j] = params_to_chi2(theta,H0_true, zcmb, zhel, Cinv, mb)
chies_norm = chies
mins = np.where(chies_norm==np.min(chies_norm))
chies_norm[mins[0][0],mins[1][0]]
omegas[mins[0][0]]
ms[mins[1][0]]
#%%
%matplotlib qt5
plt.close()
plt.matshow(np.exp(-0.5 * chies_norm));
plt.colorbar()
plt.show()
#%%
plt.figure()
chi_ms = chies_norm.sum(axis=0) #me quedo con el j
plt.plot(ms,chi_ms)
#%%
plt.figure()
chi_omegas = chies_norm.sum(axis=1)
plt.plot(omegas,chi_omegas)
#%%
len(chi_omegas)
len(chi_ms)
print(len(zcmb))
#%%
plt.figure()
chi_omegas = chies_norm[:,mins[1][0]] #me quedo con el j
plt.plot(omegas,chi_omegas)
len(chi_ms)
