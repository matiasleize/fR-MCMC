"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
np.random.seed(42)
from scipy.optimize import minimize

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO,leer_data_AGN
from alternativos import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.352
omega_m_true = 0.22
b_true = 0.023
H0_true =  70.87 #73.48 #Unidades de (km/seg)/Mpc

params_fijos = _

#%%
# Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

# Cronómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
ds_CC = leer_data_cronometros('datos_cronometros.txt')

# BAO
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
ds_BAO = []
archivos_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivos_BAO[i])
    ds_BAO.append(aux)

# AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
ds_AGN = leer_data_AGN('table3.dat')

#%% Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, index=4,
                                    dataset_SN = ds_SN,
                                    dataset_CC = ds_CC,
                                    dataset_BAO = ds_BAO,
                                    dataset_AGN = ds_AGN,
                                    #H0_Riess = True,
                                    model = 'HS',
                                    errores_agrandados=True
                                    )

initial = np.array([M_true,omega_m_true,b_true,H0_true])
soln = minimize(nll, initial, options = {'eps': 0.01}, bounds =((-25,-18),(0.1,0.5),(0, 3),(68,75)))
M_ml, omega_m_ml, b_ml, H0_ml = soln.x

print(M_ml,omega_m_ml,b_ml,H0_ml)


os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_CC+SN+BAO+AGN_4params_v2', sol=soln.x)

num_data_CC = len(ds_CC[0])
num_data_SN = len(ds_SN[0])
num_data_BAO = 20
num_data_AGN = len(ds_AGN[0])
datos_totales = num_data_CC+num_data_SN+num_data_BAO+num_data_AGN

soln.fun/(datos_totales-len(soln.x)) #

#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_CC+SN+BAO+AGN_4params_v2.npz') as data:
    sol = data['sol']

sol
