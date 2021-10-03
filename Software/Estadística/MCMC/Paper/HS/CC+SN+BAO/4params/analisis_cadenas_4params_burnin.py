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
sample = reader.get_chain()
len(sample)
burnin= burnin=int(0.2*len(sample[:,0]))
thin = int(0.5 * np.min(tau))

#%% Saving the array in a text file without thin and burnin
flat_samples = reader.get_chain(discard=0, flat=True, thin=1)
np.savez('/home/matias/Desktop/HS_CC+SN+BAO_bs.npz', bs=flat_samples[:,2])
with np.load('/home/matias/Desktop/HS_CC+SN+BAO_bs.npz') as data:
    bs = data['bs']
bs
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$', '$\Omega_{m}^{\Lambda CDM}$','b','$H_{0}^{\Lambda CDM}$'],
                    'HS (SnIA + CC + BAO)')

analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
 #%%
analisis.graficar_cadenas()
analisis.reportar_intervalos()
