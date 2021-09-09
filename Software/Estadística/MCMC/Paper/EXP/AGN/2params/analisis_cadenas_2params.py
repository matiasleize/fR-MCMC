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

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/EXP')
filename = "sample_EXP_AGN_2params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#%%
%matplotlib qt5
burnin = 500
thin = 20
analisis = Graficador(reader, ['$\Omega_{m}$','b'], 'AGN EXP')
analisis.graficar_cadenas()
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False, color='k')
analisis.reportar_intervalos()
