"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
from scipy.optimize import minimize

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon
from funciones_LambdaCDM_1 import params_to_chi2
#%%

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
#%%
#Parametros a ajustar
M_true = -19.25
omega_m_true = 0.287
H0_true =  73.48 #Unidades de (km/seg)/Mpc
#%%
#np.random.seed(42)
nll = lambda theta: params_to_chi2(theta,[M_true,H0_true], zcmb,
                    zhel, Cinv, mb)
initial = omega_m_true
#bnds = (0.01, 0.99)
soln = minimize(nll, initial,bounds=None ,options = {'eps': 0.001})
omega_m_ml = soln.x

print(omega_m_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_supernovas_LCDM_1param', sol=soln.x)
