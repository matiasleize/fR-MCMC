"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
np.random.seed(42)
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
from data import leer_data_AGN
from alternativos import params_to_chi2
#Parameters order: omega_m,b,H_0,l,n,beta,n_agn

#%% Predeterminados:
omega_m_true = 0.3
H0_true =  73.48 #Unidades de (km/seg)/Mpc

params_fijos = [_ ,_ , H0_true]

#Datos de AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
data_agn = leer_data_AGN('table3.dat')

#%% Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, dataset_AGN=data_agn,
                                    index=1, model = 'LCDM')

initial = omega_m_true
soln = minimize(nll, initial, options = {'eps': 0.01})
omega_m_ml = soln.x

print(omega_m_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_LCDM_AGN_1params', sol=soln.x)
soln.fun / (len(data_agn[0])-1)
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_LCDM_AGN_1params.npz') as data:
    sol = data['sol']

sol
