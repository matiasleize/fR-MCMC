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
from funciones_AGN import params_to_chi2_AGN_nuisance

np.random.seed(1)
#%%
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')
#%% Predeterminados:
omega_m_true = 0.9
b_true = 0.1
beta_true = 7
gamma_true = 0.6
delta_true = 0.2
H0_true =  70 #Unidades de (km/seg)/Mpc

nll = lambda theta: params_to_chi2_AGN_nuisance(theta, H0_true, data_agn, model='EXP')
initial = np.array([omega_m_true,b_true,beta_true,gamma_true,delta_true])
bnds = ((0.85,0.99),(0,0.9),(6.5,7.5),(0.5,0.8),(0.1,0.3))
soln = minimize(nll, initial, bounds=bnds, options = {'eps': 0.01})
omega_m_ml, b_ml, beta_ml, gamma_ml, delta_ml = soln.x
print(omega_m_ml,b_ml,beta_ml, gamma_ml, delta_ml)
soln.fun
print(soln.fun/(len(data_agn[0])-5))

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_EXP_AGN_5params_nuisance', sol=soln.x)
