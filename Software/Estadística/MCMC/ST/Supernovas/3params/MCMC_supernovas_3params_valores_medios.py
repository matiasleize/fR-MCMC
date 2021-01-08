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
from funciones_data import leer_data_pantheon
from funciones_supernovas import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.25
omega_m_true = 0.3
b_true = 0.1
H0_true =  73.48 #Unidades de (km/seg)/Mpc
n = 2

params_fijos = [H0_true,n]

#Datos de Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')


#%%
#Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv,mb)

initial = np.array([M_true,omega_m_true,b_true])
soln = minimize(nll, initial, options = {'eps': 0.01},
                bounds =((-19.5,-19),(0.25,0.35),(0,0.5)))
M_ml,omega_m_ml,b_ml = soln.x

print(M_ml,omega_m_ml,b_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_ST_SN_3params', sol=soln.x)
soln.fun/(len(zcmb)-3) #0.9830091200848468
#Tardo 45 min
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_ST_SN_3params.npz') as data:
    sol = data['sol']

sol
