"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""
import numpy as np
np.random.seed(42)
import emcee
import time

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_AGN
from funciones_alternativos import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

def MCMC_sampler(log_probability, initial_values,
                filename = "default.h5",
                witness_file = 'witness.txt',
                max_samples = 10000,
                witness_freq = 100,
                tolerance = 0.01):

    nwalkers, ndim = initial_values.shape

    # Set up the backend
    os.chdir(path_datos_global+'/Resultados_cadenas/')
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
    textfile_witness = open(witness_file,'w+')
    textfile_witness.close()
    #%%
    #Initialize the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
	        moves=emcee.moves.DEMove())

    # This will be useful to testing convergence
    old_tau = np.inf
    t1 = time.time()
    # Now we'll sample for up to max_samples steps
    for sample in sampler.sample(initial_values, iterations=max_samples, progress=True):
        # Only check convergence every 'witness_freq' steps
        if sampler.iteration % witness_freq: #'witness_freq' es cada cuanto chequea convergencia
            continue

        os.chdir(path_datos_global+'/Resultados_cadenas/')
        textfile_witness = open(witness_file,'w')
        textfile_witness.write('Número de iteración: {} \t'.format(sampler.iteration))

        t2 = time.time()
        textfile_witness.write('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
              int((t2-t1) - 60*int((t2-t1)/60))))
        textfile_witness.close()

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
        #También pido que tau se mantenga relativamente constante:
        converged &= np.all((np.abs(old_tau - tau) / tau) < tolerance)
        if converged:
            textfile_witness = open(witness_file,'a')
            textfile_witness.write('Convergió!')
            textfile_witness.close()
            break
        old_tau = tau

#%%
## Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

# Cronómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
ds_CC = leer_data_cronometros('datos_cronometros.txt')

# AGN
#os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
#ds_AGN = leer_data_AGN('table3.dat')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_LCDM_CC+SN_4params.npz') as data:
    sol = data['sol']
print(sol)


#Parametros fijos
params_fijos = 0

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos, index=32,
                                                        dataset_SN = ds_SN,
                                                        dataset_CC = ds_CC,
                                                        #dataset_BAO = ds_BAO,
                                                        #dataset_AGN = ds_AGN,
                                                        #H0_Riess = True,
                                                        model = 'LCDM'
                                                        )
#%%
# Definimos la distribucion del prior
def log_prior(theta):
    M, omega_m, H0 = theta
    if (-22 < M < -18 and  0.01 < omega_m < 0.5 and 60 < H0 < 80):
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
#pos = sol + 1e-4 * np.random.randn(12, 3)
pos = sol * (1 +  0.01 * np.random.randn(12,3))
MCMC_sampler(log_probability,pos,
            filename = "sample_LCDM_CC+SN_4params_m3.h5",
            witness_file = 'witness_16_m3.txt',
            witness_freq = 10,
            max_samples = 2000000)
