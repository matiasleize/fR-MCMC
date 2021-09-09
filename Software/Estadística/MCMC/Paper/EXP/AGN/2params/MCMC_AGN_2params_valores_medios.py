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
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_AGN
from funciones_alternativos import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,l,n,beta,n_agn

#Datos de AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
ds_AGN = leer_data_AGN('table3.dat')


#%% Predeterminados:
#Parametros fijos
omega_m_true = 0.3
b_true = 1

H0_true =  73.48 #Unidades de (km/seg)/Mpc

params_fijos = [_, H0_true]

#%% Parametros a ajustar

nll = lambda theta: params_to_chi2(theta, params_fijos, index=21,
                            #dataset_SN = ds_SN,
                            #dataset_CC = ds_CC,
                            #dataset_BAO = ds_BAO,
                            dataset_AGN = ds_AGN,
                            #H0_Riess = True,
                            model = 'EXP'
                            )

initial = np.array([omega_m_true,b_true])
soln = minimize(nll, initial, options = {'eps': 0.001},
                bounds =((0.2,0.5),(0,3)))
omega_m_ml,b_ml = soln.x

print(omega_m_ml,b_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_EXP_AGN_2params.npz', sol=soln.x)
soln.fun / (len(data_agn[0])-2)
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_EXP_AGN_2params.npz') as data:
    sol = data['sol']

sol
