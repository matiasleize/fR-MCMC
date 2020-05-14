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

with np.load('valores_medios_supernovas_LCDM.npz') as data:
#with np.load('valores_medios_cronom_LCDM_2params.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas/LDCM')
filename = "sample_supernovas_LCDM_M_omega_1.h5"
#filename = "sample_cronometros_LCDM_M_omega_1.h5"

reader = emcee.backends.HDFBackend(filename)
# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(tau)
#%%
%matplotlib qt5
graficar_cadenas(reader,
                labels = ['M_abs','omega_m'])
#                labels = ['omega_m', 'H0'])
 #%%
#burnin=300
graficar_contornos(reader,params_truths=sol,discard=burnin,thin=thin,
                    labels= ['M_abs','omega_m'])
#                    labels = ['omega_m', 'H0'])
#%%
#Ojo, siempre muestra que convergio, aun cuando no
plt.figure()
graficar_taus_vs_n(reader,num_param=0,threshold=1000)
graficar_taus_vs_n(reader,num_param=1,threshold=1000)
