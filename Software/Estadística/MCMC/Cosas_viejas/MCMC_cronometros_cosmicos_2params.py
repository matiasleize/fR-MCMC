"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
#!/usr/bin/env
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(42)

from scipy.optimize import minimize
import emcee
import corner
from scipy.interpolate import interp1d
import time


import sys
import os
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_cronometros
from funciones_cronometros import  params_to_chi2_H0_fijo

#%% Predeterminados:
H_0 =  73.48 #Unidades de (km/seg)/Mpc
n = 1
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
#%%
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')
omega_m_true = 0.26
b_true = 2
log_likelihood = lambda theta: -0.5 * params_to_chi2_H0_fijo(ci,theta, [H_0,n],z_data,H_data,dH)
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_cronom_2params.npz') as data:
    sol = data['sol']
omega_m_ml = sol[0]
b_ml = sol[1]

#%%
def log_prior(theta):
    omega_m, b = theta
    if 0.2 < omega_m < 0.4 and 0.5 < b < 5:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)
#%%
# Initialize the walkers
pos = sol + 1e-4 * np.random.randn(12, 2)#Posicion inicial de cada caminante
nwalkers, ndim = pos.shape

# Set up the backend
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_cron_omega_b_1.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness.txt','w+')
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
    textfile_witness = open('witness.txt','w')
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
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        textfile_witness = open('witness.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
