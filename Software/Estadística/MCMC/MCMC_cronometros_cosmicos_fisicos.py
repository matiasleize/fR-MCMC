"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
import sys
import os
from os.path import join as osjoin
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos

path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import emcee
import corner

from funciones_int import integrador
from funciones_data import leer_data_cronometros
from funciones_cronometros import chi_2_cronometros

import sys
import os
from os.path import join as osjoin
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
path_datos_global = '/home/matias/Documents/Tesis/'
#%% Predeterminados:
H_0 =  73.48 #Unidades de km/seg/Mpc
n = 1

#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1+x_0+y_0-v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales

zi = 0
zf = 3 # Es un valor razonable con las SN 1A

#%%
def params_fisicos_to_modelo(omega_m, b,n=1):
    '''Toma los parametros fisicos (omega_m y el parametro de distorsión b)
    y devuelve los parametros del modelo c1, c2 y R_HS'''
    h0 = H_0/100
    alpha = ((c_luz/1000)/8315)**2
    beta = 1 / 0.13
    aux = 6 * (1 - omega_m) / (alpha * (omega_m - beta * h0 ** 2))
    c_2 = (2/aux * b) ** n
    c_1 = aux * c_2
    r_hs = 6*(1-omega_m)*c_2/c_1
    return c_1, c_2, r_hs

#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

def params_to_chi2(theta,params_fijos):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    #[omega_m,b] = theta
    [c1,c2,r_hs] = theta
    [r_0,n] = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    #c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)
    params_modelo = [c1,r_hs,c2,n,'HS'] #de la cruz: [b,c,d,r_0,n]

    z,E = integrador(ci, params_modelo)
    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi

b_true = 0.22
omega_m_true = 0.26

#%%
c1,c2,r_hs = params_fisicos_to_modelo(omega_m_true,b_true)
print(c1,c2,r_hs)
#%%

#c1_true = 1.25*10**(-3)
#r_hs_true = 6.56*10**(-5)
#c2_true = 0.24 #c = 6*(1-omega_m)*d/b



c1_true = 1
r_hs_true = 0.24
c2_true = 1/19
np.random.seed(42)

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos=[r_0,n])
nll = lambda *args: -log_likelihood(*args)
#initial = np.array([omega_m_true,b_true]) + 0.1 * np.random.randn(2)
initial = np.array([c1_true,r_hs_true,c2_true]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial)#,bounds =([0,1],[-1,1]))
#omega_m_ml, b_ml = soln.x
c1_ml, rhs_ml, c2_ml = soln.x
#print(omega_m_ml,b_ml)
print(c1_ml,rhs_ml,c2_ml)
#%%
def log_prior(theta):
    omega_m, b = theta
    if 0 < omega_m < 1 and -1 < b < 1:
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)
pos = soln.x + 1e-4 * np.random.randn(12, 2)
nwalkers, ndim = pos.shape
#%%
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 250, progress=True);
#%%
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
#%%
tau = sampler.get_autocorr_time()
print(tau)
#%%
flat_samples = sampler.get_chain(discard=30, flat=True)#,thin=50)
print(flat_samples.shape)
#%%
fig = corner.corner(flat_samples, labels=labels, truths=[omega_m_ml,b_ml]);
#%%Guardamos los datos

os.chdir(path_git)
np.savez('Software/Estadística/Resultados_simulaciones//MCMC cronometros/samp_cron_om_b'
         , samples = samples, sampler=sampler)
