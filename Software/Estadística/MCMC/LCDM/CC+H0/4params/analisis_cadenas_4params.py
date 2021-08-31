import numpy as np
from matplotlib import pyplot as plt
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/Clases')
from funciones_graficador import Graficador

#%% Importo los mínimos del chi2
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
with np.load('valores_medios_LCDM_CC+H0_3params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_LCDM_CC+H0_3params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
#%%
burnin=100
thin=5
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','$H_{0}$'],'CC+H0 (LCDM)')
                    #'Supernovas tipo IA + Cronómetros Cósmicos + BAO')
analisis.graficar_contornos(sol, discard=burnin, thin=thin, poster=False,color='r')
#%%
analisis.graficar_cadenas()
analisis.reportar_intervalos(sol)
