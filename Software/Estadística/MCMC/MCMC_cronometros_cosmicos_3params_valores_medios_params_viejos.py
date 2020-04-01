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
from funciones_cronometros import  params_to_chi2_viejos

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

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')


c1_true = 1
c2_true = 1/19 #0.05263
c0_true = 0.24
H0_true=73.48
#np.random.seed(4)
nll = lambda theta: params_to_chi2_viejos(ci,theta,n,z_data,H_data,dH)
initial = np.array([c1_true,c2_true,c0_true,H0_true])
#bnds = ((0.01, 1), (-2, 1),(60,80))
bnds = ((0.5, 10), (0.001, 0.1), (0.1,0.3), (68,80))
soln = minimize(nll, initial,bounds=bnds)#, options = {'eps': 0.001})
#soln = minimize(nll, initial)#, options = {'eps': 0.001})
c1_ml, c2_ml, c0_ml, H0_ml = soln.x
print(c1_ml, c2_ml, c0_ml, H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_cronom_3params', sol=soln.x)
