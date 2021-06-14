"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""
import numpy as np
np.random.seed(42)
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
from funciones_data import leer_data_AGN
from funciones_AGN import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,l,n,beta,n_agn

#%% Predeterminados:
omega_m_true = 0.3
b_true = 0.1
l_true = 12 #Unidades de pc
beta_true = 0
n_agn_true = 0

H0_true =  73.48 #Unidades de (km/seg)/Mpc
n = 1


params_fijos = [H0_true,n]

#Datos de AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
z_data, Theta_data, dTheta, Sobs, alpha = leer_data_AGN('datosagn_less.dat')
dSobs = np.zeros(len(z_data))
#%%
#Parametros a ajustar
nll = lambda theta: params_to_chi2(theta, params_fijos, z_data,Theta_data,
                    dTheta,Sobs,dSobs,alpha)

initial = np.array([omega_m_true,b_true,l_true,beta_true,n_agn_true])
soln = minimize(nll, initial, options = {'eps': 0.0001},
                bounds =((0.1,0.7),(0,0.5),(1,100),(-0.1,0.6),(-1.5,0.4)))
omega_m_ml,b_ml,l_ml,beta_ml,n_agn_ml = soln.x

print(omega_m_ml,b_ml,l_ml,beta_ml,n_agn_ml)

os.chdir(path_git + '/Software/Estadística/Resultados_simulaciones')
np.savez('valores_medios_HS_AGN_5params_2', sol=soln.x)
soln.fun / (len(Theta_data)-5)
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_AGN_5params_2.npz') as data:
    sol = data['sol']
