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

b_true = 2
omega_m_true = 0.26

np.random.seed(42)
log_likelihood = lambda theta: -0.5 * params_to_chi2_H0_fijo(ci,theta, [H_0,n],z_data,H_data,dH)

os.chdir(path_git)
sys.path.append('./Software/Estadística/Resultados_simulaciones')

with np.load('valores_medios_cronom_2params.npz') as data:
    sol = data['sol']
omega_m_ml = sol[1]
b_ml = sol[0]
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
pos = sol + 1e-4 * np.random.randn(12, 2)#Posicion inicial de cada caminante
nwalkers, ndim = pos.shape
# Set up the backend
# Don't forget to clear it in case the file already exists
os.chdir(path_datos_global)
sys.path.append('./Resultados_cadenas')
filename = "sample_cron_b_omega_1.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
#%%
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
max_n = 10000
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)
# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    os.chdir(path_git)
    sys.path.append('./Software/Estadística/Resultados_simulaciones')
    np.savez('taus_cron_2params', taus=autocorr, indice = index )
    index += 1
    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
#%%
#Grafico de la autocorrelación en función del largo de la cadena.
n = 100 * np.arange(1, index + 1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");

#%% Grafico de las cadenas
#%matplotlib qt5
plt.close()
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ['omega_m','b']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
#%% Grafico de los contornos de confianza
flat_samples = sampler.get_chain(discard=20, flat=True)#,thin=50)
print(flat_samples.shape)
fig = corner.corner(flat_samples, labels=labels, truths=[omega_m_ml,b_ml]);
