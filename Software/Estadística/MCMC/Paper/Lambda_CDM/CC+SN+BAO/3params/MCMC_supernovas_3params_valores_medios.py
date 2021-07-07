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
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO
from funciones_alternativos import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.352
omega_m_true = 0.22
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

#%% Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, index=32,
                                    dataset_SN = ds_SN,
                                    dataset_CC = ds_CC,
                                    dataset_BAO = ds_BAO,
                                    #dataset_AGN = ds_AGN,
                                    #H0_Riess = True,
                                    model = 'LCDM'
                                    )

initial = np.array([M_true,omega_m_true,H0_true])
soln = minimize(nll, initial, options = {'eps': 0.01}, bounds =((-25,-18),(0.1,0.5),(68,75)))
M_ml, omega_m_ml, H0_ml = soln.x

print(M_ml,omega_m_ml,H0_ml)#-19.351100617405038 0.30819459447582237 69.2229987565787


os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_LCDM_CC+SN+BAO_3params', sol=soln.x)

num_data_CC = len(ds_CC[0])
num_data_SN = len(ds_SN[0])
num_data_BAO = 20
datos_totales = num_data_CC+num_data_SN+num_data_BAO

soln.fun/(datos_totales-len(soln.x)) #0.9910857551005845
