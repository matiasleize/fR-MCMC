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
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/LCDM')
filename = "sample_LCDM_CC+SN_4params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
samples = reader.get_chain()
burnin= int(0.2*len(samples[:,0])) #Burnin del 20%
thin = 1
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','$H_{0}$'],
                    'SnIA + CC')

analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
plt.savefig( '/home/matias/contornos_SN+CC+LCDM.png')
#%%
analisis.graficar_cadenas()
analisis.reportar_intervalos(discard=burnin, thin=thin)
