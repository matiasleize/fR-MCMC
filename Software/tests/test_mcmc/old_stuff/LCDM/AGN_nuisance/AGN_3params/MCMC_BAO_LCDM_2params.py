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
from data import leer_data_AGN
from LambdaCDM import params_to_chi2_AGN_nuisance

os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/LCDM')
with np.load('valores_medios_LCDM_AGN_3params_nuisance.npz') as data:
    sol = data['sol']
#sol[0] = 0.4
#sol[1] = 0.4
#sol[2] = 0.4
print(sol)
beta_true = 8.3
H0_true =  70
params_fijos = [H0_true, beta_true]

log_likelihood = lambda theta: -0.5 * params_to_chi2_AGN_nuisance(theta, params_fijos, data_agn)
#%% Definimos las funciones de prior y el posterior
def log_prior(theta):
    omega_m, gamma, delta = theta
    if (0.1 < omega_m < 0.99
        and 0.5 < gamma < 0.8 and 0 < delta < 0.5):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

pos = sol + 1e-4 * np.random.randn(12, 3) #Defino la cantidad de caminantes.
nwalkers, ndim = pos.shape

#%%
# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/LDCM')
filename = "sample_LCDM_AGN_3params_nuisance.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness.txt','w+')
textfile_witness.close()


#%%
#Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend
        ,moves=[(emcee.moves.DEMove(), 0.3), (emcee.moves.DESnookerMove(), 0.3)
                , (emcee.moves.KDEMove(), 0.4)])
max_n = 10000
# This will be useful to testing convergence
old_tau = np.inf


# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps (o 50)
    if sampler.iteration % 20: #20 es cada cuanto chequea convergencia
        continue

    #os.chdir(path_datos_global+'/Resultados_cadenas/')
    textfile_witness = open('witness.txt','w')
    textfile_witness.write('Número de iteración: {} \t'.format(sampler.iteration))
    textfile_witness.write('Tiempo: {}'.format(time.time()))
    textfile_witness.close()

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)

    # Check convergence
    converged_1 = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
    #También pido que tau se mantenga relativamente constante:
    converged_2 = np.all((np.abs(old_tau - tau)) < 0.001)
    if (converged_1 and converged_2):
        textfile_witness = open('witness.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
