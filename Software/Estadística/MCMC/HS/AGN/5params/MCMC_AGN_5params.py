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
from funciones_data import leer_data_AGN
from funciones_AGN import params_to_chi2
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%%
#Datos de AGN
os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
z_data, Theta_data, dTheta, Sobs, alpha = leer_data_AGN('datosagn_less.dat')
dSobs=np.zeros(len(z_data))
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_AGN_5params.npz') as data:
    sol = data['sol']
print(sol)
#Parametros fijos

H0 = 73.48
n = 1

params_fijos = [H0, n]

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos,
                z_data,Theta_data,dTheta,Sobs,dSobs,alpha)

#%% Definimos las gunciones de prior y el posterior
def log_prior(theta):
    omega_m, b, l, beta, n_agn = theta
    if not (0.1 < omega_m < 0.5 and 0 < b < 0.6
        and 5 < l < 60 and -0.01 < beta < 0.6 and -1.5 < n_agn < 0.4):
        return -np.inf

    mu_l = 11.04
    sigma_l = 1.62
    gauss_l = np.log(1.0/(np.sqrt(2*np.pi)*sigma_l))-0.5*(l-mu_l)**2/sigma_l**2

    mu_beta = -0.0024
    sigma_beta = 0.05
    gauss_beta = np.log(1.0/(np.sqrt(2*np.pi)*sigma_beta))-0.5*(beta-mu_beta)**2/sigma_beta**2

    mu_n = -0.015
    sigma_n = 0.143
    gauss_n = np.log(1.0/(np.sqrt(2*np.pi)*sigma_n))-0.5*(n-mu_n)**2/sigma_n**2

    return

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): #Este if creo que está de más..
        return -np.inf
    return lp + log_likelihood(theta)


pos = sol + 1e-4 * np.random.randn(12, 5) #Defino la cantidad de caminantes.
nwalkers, ndim = pos.shape

#%%
# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_HS_AGN_5params_1.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness_5.txt','w+')
textfile_witness.close()
#%%
#Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
max_n = 200000
# This will be useful to testing convergence
old_tau = np.inf
t1 = time.time()
# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 5: #100 es cada cuanto chequea convergencia
        continue

    os.chdir(path_datos_global+'/Resultados_cadenas/')
    textfile_witness = open('witness_5.txt','w')
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
    converged &= np.all((np.abs(old_tau - tau) / tau) < 0.01)
    if converged:
        textfile_witness = open('witness_5.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
