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
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO
from funciones_cron_SN_BAO import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
M_true = -19.352
omega_m_true = 0.24
b_true = 0.05
H0_true =  69 #Unidades de (km/seg)/Mpc
n = 1

params_fijos = [M_true,n]

#%%
#Datos de SN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

#Datos de crnómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

#Datos de BAO
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
dataset = []
archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivo_BAO[i])
    dataset.append(aux)

#%% Parametros a ajustar
nll = lambda theta: params_to_chi2(theta,params_fijos,zcmb, zhel,
                    Cinv,mb,z_data, H_data,
                    dH, dataset, M_fijo=True,chi_riess=False)

initial = np.array([omega_m_true,b_true,H0_true])
soln = minimize(nll, initial, options = {'eps': 0.01}, bounds =((0.1,0.5),(0, 3),(60,80)))
omega_m_ml, b_ml, H0_ml = soln.x

print(omega_m_ml,b_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_CC+SN+BAO_3params_Mfijo', sol=soln.x)
soln.fun/(17+len(z_data)+len(zcmb)+1-3)#1.76
