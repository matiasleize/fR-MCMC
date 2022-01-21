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
sys.path.append('./Software/utils/')
from funciones_data import leer_data_AGN
from funciones_LambdaCDM import params_to_chi2_AGN_nuisance

np.random.seed(1)
#%%
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')
len(data_agn[0])
#%% Predeterminados:
omega_m_true = 0.9
gamma_true = 0.55
delta_true = 0.2

H0_true =  70 #Unidades de (km/seg)/Mpc
beta_true = 8.3


nll = lambda theta: params_to_chi2_AGN_nuisance(theta, [H0_true,beta_true], data_agn)
initial = np.array([omega_m_true,gamma_true,delta_true])
bnds = ((0.7,1),(0.5,0.6),(0.1,0.5))
soln = minimize(nll, initial, bounds=bnds, options = {'eps': 0.001})
omega_m_ml,gamma_ml, delta_ml = soln.x
print(omega_m_ml, gamma_ml, delta_ml)
#0.9138337532890539 0.59633009719616 0.2340059103301194
#Chi_2 reducido: -0.02986072802900684

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
np.savez('valores_medios_LCDM_AGN_3params_nuisance', sol=soln.x)
soln.fun
soln.fun/(len(data_agn[0])-3)
