"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
np.random.seed(42)
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
sys.path.append('./Software/utils/')
from data import leer_data_pantheon_2
from supernovae import testeo_supernovae
#Parameters order: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.2
omega_m_true = 0.4
b_true = 0.5
H0_true =  73.48 #Unidades de (km/seg)/Mpc
alpha_true = 0.154
beta_true = 3.02
gamma_true = 0.053
n = 1

params_fijos = [n]

#%%
#Datos de SN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
_, zcmb, zhel, Cinv, mb0, x1, cor, hmass = leer_data_pantheon_2(
            'lcparam_full_long_zhel.txt','ancillary_g10.txt')
len(zcmb)
len(zhel)
len(hmass)
alpha_0=0.154
beta_0=3.02
gamma_0=0.053
mstep0=10.13
tau0=0.001
from matplotlib import pyplot as plt
plt.plot(hmass,np.power((1.+np.exp((mstep0-hmass)/tau0)),-1),'.')
plt.plot(hmass,np.heaviside(hmass-mstep0, 1),'.')


#%%
params_fijos = 1
#Parametros a ajustar
nll = lambda theta: testeo_supernovae(theta, params_fijos, zcmb, zhel, Cinv,
                    mb0,x1,cor,hmass)

initial = np.array([M_true,omega_m_true,b_true,H0_true,alpha_true,beta_true,gamma_true])
soln = minimize(nll, initial, options = {'eps': 0.01},
                bounds =((-20,-18),(0.1,0.5),(0, 1),(60,80),(0.1,0.3),(2, 4),(0,0.1)))
M_ml, omega_m_ml, b_ml, H0_ml,alpha_ml,beta_ml,gamma_ml = soln.x

print(M_ml,omega_m_ml,b_ml, H0_ml,alpha_ml,beta_ml,gamma_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_SN_7params', sol=soln.x)
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_SN_7params.npz') as data:
    sol = data['sol']

sol
