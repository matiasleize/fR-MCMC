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
from funciones_data import leer_data_BAO
from funciones_LambdaCDM import params_to_chi2_BAO

np.random.seed(1)
#%%
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
#os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_sin_nuevos')
dataset = []
archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivo_BAO[i])
    dataset.append(aux)

#%% Predeterminados:
omega_m_true = 0.27
H0_true =  73.48 #Unidades de (km/seg)/Mpc

nll = lambda theta: params_to_chi2_BAO(theta,_, dataset)
initial = np.array([omega_m_true,H0_true])
bnds = ((0.2,0.40),(60, 80))
soln = minimize(nll, initial, bounds=bnds, options = {'eps': 0.01})
omega_m_ml,H0_ml = soln.x
print(omega_m_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
np.savez('valores_medios_LCDM_BAO_2params', sol=soln.x)

# Sin usar los datos nuevos:
#soln.fun/(16-2)
#0.30232873496078727 62.80559404765493
#1.6561840628137035

#Con rd de CAMB
#0.30139137766760904 66.00279170531452
#1.6589896884098443


#Usando los datos nuevos:
soln.fun/(20-2)
#0.3114687085219484 63.011748551233964
#1.4590800745751638

#Con rd de CAMB
#0.3141531635049299 66.43172990449253
#1.4569436887471838

#%%
os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
with np.load('valores_medios_LCDM_BAO_2params.npz') as data:
    sol = data['sol']
sol
