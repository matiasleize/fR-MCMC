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
filename = "sample_HS_CC+SN_4params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

sample = reader.get_chain()
burnin= burnin=int(0.2*len(sample[:,0]))

#%%
%matplotlib qt5
#burnin=100
#thin=1
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}^{\Lambda CDM}$','b','$H_{0}^{\Lambda CDM}$'],'SNIA + CC (HS)')
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
#%%
analisis.graficar_cadenas()
analisis.reportar_intervalos(discard=burnin, thin=thin)
