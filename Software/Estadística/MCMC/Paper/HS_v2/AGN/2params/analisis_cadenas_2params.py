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
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/')
with np.load('valores_medios_HS_AGN_2params_v2.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = "sample_HS_AGN_2params_v2.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#%%
%matplotlib qt5
burnin = 1500
thin = 50
analisis = Graficador(reader, ['$\Omega_{m}$','b'], 'AGN HS')
analisis.graficar_cadenas()
analisis.graficar_contornos(sol, discard=burnin, thin=thin, poster=False, color='k')
analisis.reportar_intervalos(sol)
