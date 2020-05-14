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

with np.load('valores_medios_supernovas_LCDM_1param.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas/LDCM')
filename = "sample_supernovas_LCDM_omega_1.h5"
#filename = "sample_cronometros_LCDM_M_omega_1.h5"

reader = emcee.backends.HDFBackend(filename)
# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(tau)

#%%
#Grafico la cadena de Omega
fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
samples = reader.get_chain()
labels = ["omega_m"]
ax.plot(samples[:, :, 0], "k", alpha=0.3);

#%%
%matplotlib qt5
chain = reader.get_chain(discard=burnin)[:, :, 0].T
plt.hist(chain.flatten(), 50)
plt.gca().set_yticks([])
plt.xlabel(r"$\Omega_{m}$")
plt.ylabel(r"$p(\Omega_{m})$");

 #%%
#Ojo, siempre muestra que convergio, aun cuando no
plt.figure()
graficar_taus_vs_n(reader,num_param=0,threshold=100)
