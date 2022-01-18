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
with np.load('valores_medios_LCDM_AGN_1params.npz') as data:
    sol = data['sol']
print(sol)
#Parametros fijos
H0 = 73.48

params_fijos = [_,_,H0]

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos,
                                            dataset_AGN=data_agn, index=1,
                                            model='LCDM')

#%%
# Definimos la distribucion del prior
def log_prior(theta):
    omega_m = theta
    if 0.1 < omega_m < 0.5:
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
pos = sol + 1e-2 * np.random.randn(20, 1)

MCMC_sampler(log_probability,pos,
            filename = "sample_LCDM_AGN_1params.h5",
            witness_file = 'witness_11.txt',
            witness_freq = 5,
            max_samples = 100000)
