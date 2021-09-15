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
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/HS')
filename = "sample_HS_CC+SN+BAO_4params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
#burnin = int(2 * np.max(tau))
burnin= 0.2 * long de las cadenas
thin = int(0.5 * np.min(tau))
burnin
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$', '$\Omega_{m}^{\Lambda CDM}$','b','$H_{0}^{\Lambda CDM}$'],
                    'HS (SnIA + CC + BAO)')

analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
 #%%
analisis.graficar_cadenas()
analisis.reportar_intervalos()
