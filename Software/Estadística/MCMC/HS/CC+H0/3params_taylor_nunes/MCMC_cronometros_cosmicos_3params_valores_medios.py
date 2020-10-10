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
from funciones_cronometros import  params_to_chi2

#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n
#%% Predeterminados:
n = 1

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros_nunes.txt')

omega_m_true = 0.23
b_true = 0.1
H0_true =  73.48 #Unidades de (km/seg)/Mpc

nll = lambda theta: params_to_chi2(theta,n,z_data,H_data,dH,nunes=True,taylor=True)
initial = np.array([omega_m_true,b_true,H0_true])
bnds = ((0.05, 0.5), (-1,1), (60,100))
soln = minimize(nll, initial,bounds=bnds, options = {'eps': 0.001})
omega_m_ml, b_ml, H0_ml = soln.x
print(omega_m_ml,b_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_CC+H0_3params_taylor_nunes', sol=soln.x)
