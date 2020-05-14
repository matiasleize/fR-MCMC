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
from funciones_data import leer_data_cronometros
from funciones_cronometros import  params_to_chi2
from funciones_LambdaCDM import params_to_chi2_cronometros
#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

omega_m_true = 0.22
#H0_true =  73.48 #Unidades de (km/seg)/Mpc
H0_true =  69.01 #Unidades de (km/seg)/Mpc

#np.random.seed(4)
nll = lambda theta: params_to_chi2_cronometros(theta,z_data,H_data,dH)
initial = np.array([omega_m_true,H0_true])
bnds = ((0, 0.7), (60,80))
soln = minimize(nll, initial,bounds=None, options = {'eps': 0.001})
#soln = minimize(nll, initial)#, options = {'eps': 0.001})
omega_m_ml, H0_ml = soln.x
print(omega_m_ml,H0_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_cronom_LCDM_2params', sol=soln.x)
