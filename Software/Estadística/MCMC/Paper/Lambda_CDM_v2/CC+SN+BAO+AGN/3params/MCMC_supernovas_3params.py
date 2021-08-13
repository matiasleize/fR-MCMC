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
sys.path.append('./Software/Funcionales/')
from funciones_sampleo import MCMC_sampler
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN
from funciones_alternativos import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%%
## Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

# Cronómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
ds_CC = leer_data_cronometros('datos_cronometros.txt')

# BAO
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
ds_BAO = []
archivos_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivos_BAO[i])
    ds_BAO.append(aux)

# AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
ds_AGN = leer_data_AGN('table3.dat')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_LCDM_CC+SN+BAO+AGN_3params_v2.npz') as data:
    sol = data['sol']
print(sol)


#Parametros fijos
params_fijos = _

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos, index=32,
                                                        dataset_SN = ds_SN,
                                                        dataset_CC = ds_CC,
                                                        dataset_BAO = ds_BAO,
                                                        dataset_AGN = ds_AGN,
                                                        #H0_Riess = True,
                                                        model = 'LCDM',
                                                        errores_agrandados=True
                                                        )
#%%
# Definimos la distribucion del prior
def log_prior(theta):
    M, omega_m, H0 = theta
    if (-22 < M < -18 and  0.01 < omega_m < 0.4 and 60 < H0 < 80):
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
pos = sol + 1e-4 * np.random.randn(12, 3)

MCMC_sampler(log_probability,pos,
            filename = "sample_LCDM_CC+SN+BAO+AGN_3params_v2.h5",
            witness_file = 'witness_15_v2.txt',
            witness_freq = 5,
            max_samples = 100000)
