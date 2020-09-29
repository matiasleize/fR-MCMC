"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
np.random.seed(42)
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from scipy.interpolate import interp1d
import time

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO
from funciones_cron_SN_BAO import params_to_chi2_taylor
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#%% Predeterminados:
n = 1
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
#%%
#Datos de Supernovas
os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

#Datos de crnómetros
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

#Datos de BAO
os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
dataset = []
archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                'datos_BAO_dv.txt','datos_BAO_H.txt']
for i in range(5):
    aux = leer_data_BAO(archivo_BAO[i])
    dataset.append(aux)

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
#with np.load('valores_medios_HS_cron+SN+BAO_3params_Mfijo_taylor.npz') as data:
with np.load('valores_medios_HS_cron+SN+BAO_3params_Mfijo_sin_riess_taylor.npz') as data:
    sol = data['sol']
print(sol)

#Parametros a ajustar
M_true=-19.352
n=1
params_fijos =[M_true,n]

log_likelihood = lambda theta: -0.5 * params_to_chi2_taylor(theta,params_fijos,
                zcmb, zhel, Cinv,mb,z_data, H_data, dH, dataset,M_fijo=True,chi_riess=False)
#%% Definimos las gunciones de prior y el posterior

def log_prior(theta):
    omega_m, b, H0 = theta
    if (0.01 < omega_m < 0.5 and -3 < b < 3 and 60 < H0 < 80):
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
os.chdir(path_datos_global+'/Resultados_cadenas/')
#filename = "sample_HS_cron+SN+BAO_3params_Mfijo_taylor.h5"
filename = "sample_HS_cron+SN+BAO_3params_Mfijo_sin_riess_taylor.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
textfile_witness = open('witness_5.txt','w+')
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
    textfile_witness = open('witness_5.txt','w')
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
        textfile_witness = open('witness_1.txt','a')
        textfile_witness.write('Convergió!')
        textfile_witness.close()
        break
    old_tau = tau
