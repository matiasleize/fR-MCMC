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
min_z = 0.5
max_z = 1.1


os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt'
                                    ,min_z=min_z,max_z=max_z)

sn = len(zcmb)
#%%
#Parametros a ajustar
M_true = -19.25
#omega_m_true = 0.3089
#H0_true =  73.48 #Unidades de (km/seg)/Mpc

omegas = np.linspace(0,1,20)
hs = np.linspace(60,90,20)
len(hs)
chies = np.zeros((len(omegas),len(hs)))

for i,omega_m in enumerate(omegas):
    for j,H0 in enumerate(hs):
        theta = [omega_m,H0]
        chies[i,j] = params_to_chi2(theta,M_true, zcmb, zhel, Cinv, mb,Mfijo=True)
chies_norm = chies
log_likelihood = -0.5 * chies_norm
#print(chies_norm)
#%%
%matplotlib qt5
plt.close()
plt.matshow(np.exp(log_likelihood));
plt.colorbar()
plt.show()
#%%
plt.figure()
chi_hs = chies_norm.sum(axis=0) #me quedo con el j
plt.plot(hs,chi_hs)
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
chi_omegas = chies_norm[:,15] #me quedo con el j
plt.plot(omegas,chi_omegas)
len(chi_ms)
