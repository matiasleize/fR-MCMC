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
sys.path.append('./Software/utils/')
from sampling import MCMC_sampler
from AGN import params_to_chi2_AGN_nuisance
from data import leer_data_AGN

#%%
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
with np.load('valores_medios_LCDM_AGN_4params_nuisance_3.npz') as data:
   sol = data['sol']
print(sol)
H0_true =  73.24 #EL valor de Pantheon, pero con 70 da bastante similar!

log_likelihood = lambda theta: -0.5 * params_to_chi2_AGN_nuisance(theta,H0_true,
                                        data_agn, model='LCDM',less_z=True)

#%% Definimos la distribucion del prior
def log_prior(theta):
    omega_m, beta, gamma, delta = theta
    if not (5 < beta < 15 and 0.1 < gamma < 0.8 and 0 < delta < 0.5):
        return -np.inf
    #gaussian prior on a
    mu = 0.298
    sigma = 0.022
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(omega_m-mu)**2/sigma**2

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
            filename = "sample_LCDM_AGN_4params_nuisance_3.h5",
            witness_file = 'witness_123.txt',
            witness_freq = 5,
            max_samples = 1000000)
