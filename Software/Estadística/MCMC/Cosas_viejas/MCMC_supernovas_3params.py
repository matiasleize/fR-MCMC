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


import sys
import os
from os.path import join as osjoin
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
path_datos_global = '/home/matias/Documents/Tesis/'
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador
from funciones_data import leer_data_pantheon
from funciones_supernovas import params_to_chi2_H0_fijo

#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,b,omega_m,H_0,n


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


os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

#Parametros a ajustar
M_true = -19.6
b_true = 2
omega_m_true = 0.26
#%%


np.random.seed(42)
log_likelihood = lambda theta: -0.5 * params_to_chi2_H0_fijo(ci, theta,[H_0,n], zcmb, zhel, Cinv, mb)

nll = lambda *args: -log_likelihood(*args)
initial = np.array([M_true, b_true,omega_m_true]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial, bounds =([-19.8,-19.2],[-10,10],[0,1]))
m_ml, b_ml, omega_m_ml = soln.x

print(m_ml,b_ml,omega_m_ml)


#HASTA ACA

#%% Definimos las gunciones de prior y el posterior

def log_prior(theta):
    M, b, omega_m = theta
    if -20 < M < -19 and 0.5 < b < 2 and 0.2 < omega_m < 0.9:
        return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


pos = soln.x + 1e-4 * np.random.randn(10, 3) #Defino la cantidad de caminantes.
nwalkers, ndim = pos.shape
#%% Corremos el MonteCarlo
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 100, progress=True); #Defino la cant de pasos

#%% Graficamos las cadenas de Markov
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["M", "b","omega_m"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

#%%
flat_samples = sampler.get_chain(discard=1, flat=True)
print(flat_samples.shape)
#%% Graficamos las gausianas 2D y proyectadas
fig = corner.corner(
    flat_samples, labels=labels, truths=[M_true, b_true, omega_m_true]
);
#%% Guardado
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
np.savez('Software/supernovas_samp_M_b_omega', samples = samples, sampler=sampler)
