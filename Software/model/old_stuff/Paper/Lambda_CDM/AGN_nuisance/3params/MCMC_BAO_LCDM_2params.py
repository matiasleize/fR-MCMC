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
import time
np.random.seed(1)

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_sampleo import MCMC_sampler
from funciones_AGN import params_to_chi2_AGN_nuisance
from funciones_data import leer_data_AGN

#%%
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
with np.load('valores_medios_LCDM_AGN_4params_nuisance.npz') as data:
   sol = data['sol']
print(sol)
H0_true =  70

log_likelihood = lambda theta: -0.5 * params_to_chi2_AGN_nuisance(theta,H0_true, data_agn, model='LCDM')

#%% Definimos la distribucion del prior
def log_prior(theta):
    omega_m, beta, gamma, delta = theta
    if (0.1 < omega_m < 0.99 and 5 < beta < 15
        and 0.1 < gamma < 0.8 and 0 < delta < 0.5):
        return 0.0
    return -np.inf

# Definimos la distribución del posterior
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

#%%
#Defino los valores iniciales de cada cadena a partir de los valores
#de los parametros que corresponden al minimo del chi2.
pos = sol + 1e-2 * np.random.randn(12, 4)

MCMC_sampler(log_probability,pos,
            filename = "sample_LCDM_AGN_4params_nuisance.h5",
            witness_file = 'witness_12.txt',
            witness_freq = 5,
            max_samples = 1000000)
