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
sys.path.append('./Software/utils/')
from funciones_data import leer_data_pantheon_2
from funciones_supernovas import testeo_supernovas
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%%
#Datos de Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
_, zcmb, zhel, Cinv, mb0, x1, cor, hmass = leer_data_pantheon_2(
            'lcparam_full_long_zhel.txt','ancillary_g10.txt')

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_SN_4params.npz') as data:
    sol = data['sol']
print(sol)

#Parametros fijos
omega_m = 0.3
b = 0.5
H_0 = 73.48
n = 1
params_fijos = [omega_m,b,H_0,n]

log_likelihood = lambda theta: -0.5 * testeo_supernovas(theta, params_fijos,
                zcmb, zhel, Cinv, mb0, x1, cor, hmass)

#%% Definimos las gunciones de prior y el posterior
def log_prior(theta):
    M, alpha, beta, gamma = theta
    if (-20 < M < -18.5 and 0.15 < alpha < 0.165 and 2.95 < beta < 3.09 and
        0.047 < gamma < 0.065):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


pos = sol + 1e-4 * np.random.randn(12, 4) #Defino la cantidad de caminantes.
nwalkers, ndim = pos.shape

#%%
# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_HS_SN_4params.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness_2.txt','w+')
textfile_witness.close()
#%%
#Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
max_n = 10000
# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 5: #100 es cada cuanto chequea convergencia
        continue

    os.chdir(path_datos_global+'/Resultados_cadenas/')
    textfile_witness = open('witness_2.txt','w')
    textfile_witness.write('Número de iteración: {} \t'.format(sampler.iteration))
    textfile_witness.write('Tiempo: {}'.format(time.time()))
    textfile_witness.close()

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
    #También pido que tau se mantenga relativamente constante:
    converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01)
    if converged:
        textfile_witness = open('witness_2.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
