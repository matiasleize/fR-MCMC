import numpy as np
np.random.seed(42)
from scipy.optimize import minimize
import emcee
import time

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_sampleo import MCMC_sampler
from funciones_alternativos import params_to_chi2
from funciones_data import leer_data_AGN
#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

#%%
#Datos de AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
data_agn = leer_data_AGN('table3.dat')
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_AGN_2params.npz') as data:
    sol = data['sol']
print(sol)
#Parametros fijos
H0 = 73.48
n = 1

params_fijos = [_,H0,n]

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos,
                                            dataset_AGN=data_agn, index=1)

#%%
# Definimos la distribucion del prior
def log_prior(theta):
    omega_m, b = theta
    if (0.1 < omega_m < 0.5 and 0 < b < 1):
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
pos = sol + 1e-4 * np.random.randn(12, 2)

MCMC_sampler(log_probability,pos,
            filename = "sample_HS_AGN_2params.h5",
            witness_file = 'witness.txt',
            witness_freq = 5,
            max_samples = 100000)
