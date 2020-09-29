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
from funciones_BAO import params_to_chi2_taylor

np.random.seed(1)
#%%
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
dataset = []
archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivo_BAO[i])
    dataset.append(aux)
#%% Predeterminados:
omega_m_true = 0.24
b_true = 0.01
H0_true = 73.48 #Unidades de (km/seg)/Mpc
n = 1
#%%
nll = lambda theta: params_to_chi2_taylor(theta, [H0_true,n], dataset,model='ST')
initial = np.array([omega_m_true,b_true])
bnds = ((0.1,0.5),(-2,4))
soln = minimize(nll, initial, bounds=bnds)#, options = {'eps': 0.01})
omega_m_ml, b_ml = soln.x
print(omega_m_ml, b_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones/LCDM')
np.savez('valores_medios_ST_BAO_2params_taylor', sol=soln.x)

soln.fun/(17-2) #2.723810113885105
