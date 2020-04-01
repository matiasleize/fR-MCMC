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
from funciones_data import leer_data_pantheon
from funciones_supernovas import params_to_chi2

#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,b,omega_m,H_0,n

#%% Predeterminados:
n = 1
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
#%%

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
#%%
#Parametros a ajustar
M_true = -19.6
omega_m_true = 0.26
b_true = -1.2
H0_true =  73#73.48 #Unidades de (km/seg)/Mpc

#%%
np.random.seed(42)
nll = lambda theta: params_to_chi2(ci, theta, n, zcmb, zhel, Cinv, mb)
initial = np.array([M_true,omega_m_true,b_true,H0_true])
soln = minimize(nll, initial, bounds =((-20,-18),(0.2,0.3),(None, None),(60,80)))#,options = {'eps': 0.001})
M_ml, omega_m_ml, b_ml, H0_ml = soln.x

print(M_ml,omega_m_ml,b_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_supernovas_4params', sol=soln.x)
