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
from funciones_LambdaCDM import params_to_chi2_3params
#%%

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
#%%
#Parametros a ajustar
M_true = -19.5
omega_m_true = 0.26
H0_true =  63.48 #Unidades de (km/seg)/Mpc

#%%
#np.random.seed(42)
nll = lambda theta: params_to_chi2_3params(theta, zcmb, zhel, Cinv, mb) #CAMBIAR
initial = np.array([M_true,omega_m_true,H0_true])
soln = minimize(nll, initial, bounds =((-25,-15),(0.01,0.9),(50,80)),options = {'eps': 0.001})
M_ml, omega_m_ml, H0_ml = soln.x

print(M_ml,omega_m_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_supernovas_LCDM', sol=soln.x)
