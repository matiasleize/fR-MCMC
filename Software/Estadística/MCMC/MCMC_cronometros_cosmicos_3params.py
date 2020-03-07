"""
Created on Wed Feb  5 16:07:35 2020

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
from funciones_cronometros import params_to_chi2

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


os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

b_true = 2
omega_m_true = 0.26
H0_true = 73.48

np.random.seed(42)
log_likelihood = lambda theta: -0.5 * params_to_chi2_H0(ci,theta,n,z_data,H_data,dH)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([omega_m_true,b_true,H0_true]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial)
omega_m_ml, b_ml, H0_ml = soln.x
print(omega_m_ml,b_ml,H0_ml)

#%%
def log_prior(theta):
    omega_m, b, H_0 = theta
    if 0.1 < omega_m < 1 and 0.5 < b < 5 and 20 < H_0 < 85:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)
pos = soln.x + 1e-4 * np.random.randn(12, 3)
nwalkers, ndim = pos.shape
#%%
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 100, progress=True);
#%%
plt.close()
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['omega_m','b', 'H0']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
#%%
tau = sampler.get_autocorr_time()
print(tau)
#%%
flat_samples = sampler.get_chain(discard=1, flat=True,thin=50)
print(flat_samples.shape)
#%%
fig = corner.corner(flat_samples, labels=labels, truths=[omega_m_ml,b_ml,H_0_ml]);
#%%Guardamos los datos

os.chdir(path_git)
np.savez('Software/Estadística/Resultados_simulaciones//MCMC cronometros/samp_cron_omega_b_h0'
         , samples = samples, sampler=sampler)
