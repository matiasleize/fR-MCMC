"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
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
from funciones_data import leer_data_cronometros
from funciones_cronometros import params_to_chi2

#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n
#%% Predeterminados:
n = 1
#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')
log_likelihood = lambda theta: -0.5 * params_to_chi2(theta,n,z_data,H_data,dH)
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_CC+H0_3params.npz') as data:
    sol = data['sol']
print(sol)


#%%
def log_prior(theta):
    omega_m, b, H_0 = theta
    if (0.05 < omega_m < 0.4 and 0 < b < 5 and 63 < H_0 < 80):
        return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)
pos = sol + 1e-4 * np.random.randn(12, 3)
nwalkers, ndim = pos.shape
#%%
# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_HS_CC+H0_3params.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness_4.txt','w+')
textfile_witness.close()
#%%
#Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
        moves=[(emcee.moves.DEMove(), 0.4), (emcee.moves.DESnookerMove(), 0.3)
        , (emcee.moves.KDEMove(), 0.3)])
max_n = 10000
# This will be useful to testing convergence
old_tau = np.inf
t1 = time.time()

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 20: #100 es cada cuanto chequea convergencia
        continue

    os.chdir(path_datos_global+'/Resultados_cadenas/')
    textfile_witness = open('witness_4.txt','w')
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
    converged_1 = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
    #(threshold nunca debe ser menor que 50 según la documentación de emcee! )
    #También pido que tau se mantenga relativamente constante:
    converged_2 = np.all((np.abs(old_tau - tau)/tau) < 0.001) #Error menor a 0.1%
    if (converged_1 and converged_2):
        textfile_witness = open('witness_4.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
