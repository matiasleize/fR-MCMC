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
with np.load('valores_medios_cronom_2params.npz') as data:
    sol = data['sol']
#%%
os.chdir(path_datos_global+'/Resultados_cadenas/')
#filename = 'sample_supernovas_M_b_101.h5'
filename = 'sample_cron_omega_b_1.h5'

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
                #labels = ['M_abs','b'])
                labels = ['omega_m','b'])
#%%
burnin=300
#burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
graficar_contornos(reader,params_truths=sol,discard=burnin,#thin=thin
                    #labels= ['M_abs','b'])
                    labels = ['omega_m','b'])
#%%
plt.figure()
graficar_taus_vs_n(reader,num_param=0)
graficar_taus_vs_n(reader,num_param=1)
