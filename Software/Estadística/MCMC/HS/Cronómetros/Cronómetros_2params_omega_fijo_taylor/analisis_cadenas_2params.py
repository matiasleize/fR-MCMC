import numpy as np
import emcee
from matplotlib import pyplot as plt
import corner
import sys
import os
import time

from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_analisis_cadenas import graficar_cadenas,graficar_contornos,graficar_taus_vs_n
#%%
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_cronom_2params_omega_fijo_taylor.npz') as data:
    sol = data['sol']
sol
#%%
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_HS_cronom_2params_omega_fijo_taylor.h5"
reader = emcee.backends.HDFBackend(filename)
# Algunos valores
nwalkers, ndim = reader.shape
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(nwalkers,ndim)
print(samples.shape[0])#numero de pasos
print(tau)
#%%
%matplotlib qt5
graficar_cadenas(reader,
                labels = ['b','$H_0$'],title='CC HS (Taylor) $\Omega_m=0.24$')
#%%
#burnin=100
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
graficar_contornos(reader,params_truths=sol,discard=burnin,thin=thin,
                    labels = ['b','$H_0$'],title='CC HS (Taylor) $\Omega_m=0.24$')
#%%
plt.figure()
graficar_taus_vs_n(reader,num_param=0)
graficar_taus_vs_n(reader,num_param=1)
#%% Printeo los valores!
from IPython.display import display, Math
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
labels = ['b','$H_0$']
len_chain,nwalkers,ndim=reader.get_chain().shape
for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    mcmc[1]=sol[i] #Correción de mati: En vez de percentil 50 poner el mu
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))
