"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""
import numpy as np
np.random.seed(42)
import emcee

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from sampleo import MCMC_sampler
from data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO_odintsov
from alternativos_odintsov import params_to_chi2_odintsov
#Parameters order: Mabs,omega_m,b,H_0,n

#%%
## Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

# Cronómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
ds_CC = leer_data_cronometros('chronometers_data.txt')

# BAO de odintsov
os.chdir(path_git+'/Software/Estadística/Datos/BAO/Odintsov_data')
ds_BAO = leer_data_BAO_odintsov('BAO_data_odintsov.txt')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_EXP_CC+SN+BAO_4params_odintsov.npz') as data:
    sol = data['sol']
print(sol)


#Parametros fijos
params_fijos = 0

log_likelihood = lambda theta: -0.5 * params_to_chi2_odintsov(theta, params_fijos, index=4,
                                                        dataset_SN = ds_SN,
                                                        dataset_CC = ds_CC,
                                                        dataset_BAO = ds_BAO,
                                                        #dataset_AGN = ds_AGN,
                                                        #H0_Riess = True,
                                                        model = 'EXP'
                                                        )
#%%
# Definimos la distribucion del prior
def log_prior(theta):
    M, omega_m, b, H0 = theta
    if (-22 < M < -18 and  0.01 < omega_m < 0.5 and 0 < b < 3 and 60 < H0 < 80):
        return 0.0
    return -np.inf

# Definimos la distribución del posterior
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): #Este if creo que está de más..
        return -np.inf
    return lp + log_likelihood(theta)

#%%
#Defino los valores iniciales de cada cadena a partir de los valores
#de los parametros que corresponden al minimo del chi2.
pos = sol + 1e-4 * np.random.randn(12, 4)

MCMC_sampler(log_probability,pos,
            filename = "sample_EXP_CC+SN+BAO_4params_odintsov.h5",
            witness_file = 'witness_34_odintsov.txt',
            witness_freq = 5,
            max_samples = 2000000)
