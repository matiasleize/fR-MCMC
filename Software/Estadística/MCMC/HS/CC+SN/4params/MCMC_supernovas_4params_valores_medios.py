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
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon, leer_data_cronometros
from funciones_cron_SN import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.2
omega_m_true = 0.2
b_true = 0.3
H0_true =  73.48 #Unidades de (km/seg)/Mpc
n = 1

params_fijos = [n]

#%%
#Datos de SN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

#Datos de crnómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

#%%
params_fijos = 1
#Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv,
                    mb, z_data, H_data, dH, chi_riess=False)

initial = np.array([M_true,omega_m_true,b_true,H0_true])
soln = minimize(nll, initial, options = {'eps': 0.01},
                bounds =((-19.8,-19),(0.1,0.5),(0, 1),(68,75)))
M_ml, omega_m_ml, b_ml, H0_ml = soln.x

print(M_ml, omega_m_ml, b_ml, H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_CC+SN_4params', sol=soln.x)

soln.fun/(len(z_data)+len(zcmb)-4) #0.9709124005246293
