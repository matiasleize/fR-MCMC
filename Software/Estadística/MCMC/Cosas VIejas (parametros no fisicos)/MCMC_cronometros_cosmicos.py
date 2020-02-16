"""
Created on Wed Feb  5 16:07:35 2020

@author: matias
"""
import sys
import os
from os.path import join as osjoin
from scipy.interpolate import interp1d
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'

path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
#sys.path.append('/Software/Estadística/')


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

os.chdir(path_git)
#sys.path.append('./Software/Funcionales/')
#sys.path.append('/Software/Estadística/')

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
#Parametros a ajustar
b0 = 1
d0 = 1/19 #(valor del paper)
c0 = 0.24
n0 = 1
#%%

os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

def params_to_chi2(theta,params_fijos):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [b,c,d] = theta
    [r_0,n] = params_fijos
    
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
    H_int = interp1d(z,E)     
    H_teo = H_0 * H_int(z_data)
    
    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi

#%%
b_true = 1
c_true = 0.24
d_true = 1/19

np.random.seed(42)

log_likelihood = lambda theta: -0.5 * params_to_chi2(theta, params_fijos=[r_0,n0])
nll = lambda *args: -log_likelihood(*args)
initial = np.array([b_true,c_true,d_true]) + 0.1 * np.random.randn(3)
soln = minimize(nll, initial)#,bounds =([0.93,1.5],[0.01,0.5]))
b_ml, c_ml, d_true = soln.x
#0.9959119437266656 0.22617356988288154 6.811764744092102e-05


print(b_ml,c_ml,d_true)

#%%
def log_prior(theta):
    b, c, d = theta
    if 0 < b < 5 and 0 < c < 1 and 0 < d < 0.3:
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
sampler.run_mcmc(pos, 10000, progress=True);
#%%
plt.close()
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["b",'c','d']
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
flat_samples = sampler.get_chain(discard=10, flat=True,thin=50)
print(flat_samples.shape)
#%%
fig = corner.corner(flat_samples, labels=labels, truths=[b_true, c_true,d_true]);
#%%Guardamos los datos

os.chdir(path_git)
np.savez('Software/Estadística/Resultados_simulaciones//MCMC cronometros/samp_cron_b_c_d_10000'
         , samples = samples, sampler=sampler)

#%% prueba de integracion
b = 1
d = 1/19 #(valor del paper)
c = 0.24
n = 1

[b,c,d,r_0,n] = params_modelo

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

os.chdir(path_git+'/Software/Estadística/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')

z,E = integrador(dX_dz,ci, params_modelo)
H_int = interp1d(z,E)     
H_teo = H_0 * H_int(z_data)

z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')
#%%
plt.plot(z_data,H_data,'.')
plt.errorbar(z_data,H_data,yerr=dH, linestyle="None")
plt.plot(z_data, H_teo,'.')
#%%
#3.244860421067025 0.22617356988288154 (con 7000 y 0.01) 2min 11 seg
#3.2442388852659785 0.22617356988288154 (con 100 y 0.01) 1 seg

#3.244934859519713 0.22617356988288154 (con 2000 y 0.05)

#3.245434969720828 0.22617356988288154 (con 2000 y 0.1) 7 seg
#3.245429162903268 0.22617356988288154 (con 1000 y 0.1) 4 seg
#3.2454046248104236 0.22617356988288154 (con 500 y 0.1) 1 seg
#3.2446939875436343 0.22617356988288154 (con 100 y 0.1) 0 seg

#3.232733812367659 0.22617356988288154 (con 100 y 0.5) 0 seg
#3.2335397987408556 0.22617356988288154 (con 2000 y 0.5) 1 seg

