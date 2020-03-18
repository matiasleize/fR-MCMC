"""
Created on Wed Feb  5 16:07:35 2020

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
from funciones_data import leer_data_cronometros
from funciones_cronometros import  params_to_chi2_H0_fijo

#%% Predeterminados:
H_0 =  73.48 #Unidades de (km/seg)/Mpc
n = 1
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

b_true = 2
omega_m_true = 0.26

np.random.seed(42)
log_likelihood = lambda theta: -0.5 * params_to_chi2_H0_fijo(ci,theta, [H_0,n],z_data,H_data,dH)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([omega_m_true,b_true]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial)#,bounds =([0,1],[-10,10]))
omega_m_ml, b_ml = soln.x
print(omega_m_ml,b_ml)

os.chdir(path_git)
sys.path.append('./Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_cronom_2params', sol=soln.x )
