"""
Created on Wed Feb  5 16:07:35 2020

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
from funciones_data import leer_data_cronometros
from funciones_cronometros import  params_to_chi2_taylor

#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n
#%% Predeterminados:
n = 1

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

omega_m_true = 0.24
b_true = 0
H0_true =  73.48 #Unidades de (km/seg)/Mpc

nll = lambda theta: params_to_chi2_taylor(theta,[omega_m_true,n],z_data,H_data,dH,omega_fijo=True)
initial = np.array([b_true,H0_true])
bnds = ((-1,1), (60,80))
soln = minimize(nll, initial,bounds=bnds)#, options = {'eps': 0.01})
b_ml, H0_ml = soln.x
print(b_ml, H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_CC+H0_2params_omega_fijo_taylor', sol=soln.x)

soln.fun/(len(z_data)-2) #0.54
