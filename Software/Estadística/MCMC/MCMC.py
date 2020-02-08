#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:02:59 2020

@author: matias
"""
import sys
import os
from os.path import join as osjoin
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'

path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
#sys.path.append('/Software/Estadística/')


import numpy as np

from matplotlib import pyplot as plt

from funciones_int import integrador, magn_aparente_teorica
from funciones_data import leer_data_pantheon
from funciones_estadistica import chi_2

#%% Predeterminados:
H_0 =  73.48
#Gamma de Lucila (define el modelo)
gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))  
#Coindiciones iniciales e intervalo
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1+x_0+y_0-v_0 
r_0 = 41

ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
zi = 0
zf = 3 # Es un valor razonable con las SN 1A

n = 1
#%%
#Parametros a ajustar


def params_to_chi2(theta, params_fijos):
    '''Dados los parámetros del modelo devuelve un chi2'''

    [r_0,n] = params_fijos

    [Mabs,b,c,d] = theta 
    params_modelo = [b,c,d,r_0,n]
    def dX_dz(z, variables): 
        x = variables[0]
        y = variables[1]
        v = variables[2]
        w = variables[3]
        r = variables[4]
        
        G = gamma(r,b,c,d,n)
        
        s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
        s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
        s2 = -v * (x*G + 4 - 2*v) / (z+1)
        s3 = w * (-1 + x+ 2*v) / (z+1)
        s4 = -x*r*G/(1+z)        
        return [s0,s1,s2,s3,s4]
    
    z,E = integrador(dX_dz,ci, params_modelo)
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
    muth = magn_aparente_teorica(z,E,zhel,zcmb)
    
    if isinstance(Mabs,list):
        chis_M = np.zeros(len(Mabs))
        for i,M_0 in enumerate(Mabs):
            chis_M[i] = chi_2(muth,mb,M_0,Cinv)
        return chis_M
    else:
        chi2 = chi_2(muth,mb,Mabs,Cinv)
        return chi2
    
#%%
from scipy.optimize import minimize


M_true = 19.6
b_true = 1.1
c_true = 0.24
d_true = 1/19

np.random.seed(42)
nll = lambda theta: -0.5 * params_to_chi2(theta, params_fijos=[r_0,n])
initial = np.array([M_true, b_true,c_true,d_true]) + 0.1 * np.random.randn(4)
soln = minimize(nll, initial,bounds =([19.2,19.8],[0.95,1.2],[0.1,0.4],[0.02,0.07]))
m_ml, b_ml, c_ml, d_ml = soln.x

print(m_ml,b_ml,c_ml,d_ml)

#%%
nll = lambda theta: -0.5 * params_to_chi2(theta, params_fijos=[r_0,n])
import emcee


def log_prior(theta):
    M, b, c,d = theta
#    if 19 < M < 19.6 and 0.95 < b < 1.5 and 0.2 < c < 0.3 and 0 < d < 0.04:
    if 18 < M < 20 and 0.5 < b < 4 and 0 < c < 1 and 0 < d < 0.3:
        return 0.0
    return -np.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + nll(theta)


pos = soln.x + 1e-4 * np.random.randn(32, 4)
nwalkers, ndim = pos.shape
#%%
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, int(1e4), progress=True);

#%%
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b", "c","d"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

#%%
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
print(flat_samples.shape)

#%%
import corner

fig = corner.corner(
    flat_samples, labels=labels, truths=[M_true, b_true, c_true, d_true]
);
