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
from funciones_LambdaCDM_BAO_old_data import params_to_chi2_BAO_old_data

np.random.seed(1)
#%%
os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_viejos')
archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']

def leer_data_BAO(archivo_BAO):
        z, valores_data, errores_data, rd_fid = np.loadtxt(archivo_BAO,
        usecols=(0,1,2,4),unpack=True)
        return z, valores_data, errores_data, rd_fid

dataset_BAO = []
for i in range(5):
    aux = leer_data_BAO(archivo_BAO[i])
    dataset_BAO.append(aux)


#%% Predeterminados:
omega_m_true = 0.27
H0_true =  73.48 #Unidades de (km/seg)/Mpc


nll = lambda theta: params_to_chi2_BAO_old_data(theta, _, dataset_BAO)
initial = np.array([omega_m_true,H0_true])
bnds = ((0.2,0.40),(60, 80))
soln = minimize(nll, initial, bounds=bnds, options = {'eps': 0.001})
omega_m_ml,H0_ml = soln.x
print(omega_m_ml,H0_ml) #0.3299125386121046 66.50133421464099

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
np.savez('valores_medios_LCDM_BAO_2params_old_data', sol=soln.x)
soln.fun/(17-2) #1.2623399000133013

#%%
os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
with np.load('valores_medios_LCDM_BAO_2params_old_data.npz') as data:
    sol = data['sol']
sol
