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
from funciones_data import leer_data_pantheon
from funciones_LambdaCDM import params_to_chi2

min_z = 0
max_z = 3

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt'
                        ,min_z=min_z,max_z=max_z)

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/LCDM/')
with np.load('valores_medios_LCDM_supernovas_2params.npz') as data:
    sol = data['sol']
#%%
#Parametros a ajustar
H0_true = 73.48 #Unidades de (km/seg)/Mpc
log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, H0_true, zcmb,
                                                     zhel, Cinv, mb,fix_H0=True)

#%% Definimos las gunciones de prior y el posterior

#Prior uniforme
def log_prior(theta):
    M, omega_m = theta
    if (-20 < M < -18.5 and  0 < omega_m < 1) :
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

pos = sol + 1e-4 * np.random.randn(100, 2) #Defino la cantidad de caminantes.
nwalkers, ndim = pos.shape

#%%
# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/LDCM')
filename = "sample_LCDM_supernovas_2params.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
#textfile_witness = open('witness.txt','w+')
#textfile_witness.close()

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
    #textfile_witness = open('witness.txt','w')
    #textfile_witness.write('Número de iteración: {} \t'.format(sampler.iteration))
    #textfile_witness.write('Tiempo: {}'.format(time.time()))
    #textfile_witness.close()

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)

    # Check convergence
    converged_1 = np.all(tau * 1000 < sampler.iteration) #100 es el threshold de convergencia
    #También pido que tau se mantenga relativamente constante:
    converged_2 = np.all((np.abs(old_tau - tau)/tau) < 0.001)
    if (converged_1 and converged_2):
    #    textfile_witness = open('witness.txt','a')
    #    textfile_witness.write('Convergió!')
    #    textfile_witness.close()
        break
    old_tau = tau
