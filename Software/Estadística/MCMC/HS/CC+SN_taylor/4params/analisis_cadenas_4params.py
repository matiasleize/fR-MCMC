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
os.chdir(path_datos_global+'/Resultados_cadenas/')
filename = "sample_HS_CC+SN_4params_taylor.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain()
#burnin= int(0.2*len(samples[:,0]))
#thin=1

#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'SNIA + CC (HS) Taylor')
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
#%%
analisis.graficar_cadenas()
analisis.reportar_intervalos(hdi=True)
