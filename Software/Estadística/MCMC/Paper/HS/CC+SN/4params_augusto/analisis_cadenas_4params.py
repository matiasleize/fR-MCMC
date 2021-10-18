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
filename = "sample_HS_CC+SN_4params_augusto.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
samples = reader.get_chain()
burnin= int(0.2*len(samples[:,0]))
thin=1

# Saving the array in a text file
#np.savez('/home/matias/Desktop/HS_CC+SN_bs_20_cadenas.npz', bs=samples[:,2])
#with np.load('/home/matias/Desktop/HS_CC+SN_bs_20_cadenas.npz') as data:
#    bs = data['bs']


#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}^{\Lambda CDM}$','b','$H_{0}^{\Lambda CDM}$'],'SNIA + CC (HS) Matias 20 cadenas')
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
#analisis.graficar_cadenas()
analisis.reportar_intervalos(discard=burnin, thin=thin)
#analisis.graficar_taus_vs_n()

#%%
os.chdir(path_datos_global+'/Resultados_cadenas')
filename = 'CC+SnIa_HS_NN.h5' #Corridas de Augusto
reader_agus = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader_agus.get_autocorr_time()
samples = reader.get_chain()
burnin= int(0.2*len(samples[:,0]))
thin=1

analisis_1 = Graficador(reader_agus, ['b','$\Omega_{m}^{\Lambda CDM}$','$H_{0}^{\Lambda CDM}$','$M_{abs}$'],'SNIA + CC (HS)')
analisis_1.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
#analisis.graficar_cadenas()
analisis_1.reportar_intervalos(discard=burnin, thin=thin)
