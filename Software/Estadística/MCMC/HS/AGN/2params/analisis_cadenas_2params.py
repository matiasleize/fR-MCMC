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
with np.load('valores_medios_HS_AGN_2params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
#os.chdir(path_datos_global+'/Resultados_cadenas')
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/12 cadenas/HS')
filename = "sample_HS_AGN_2params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores

samples = reader.get_chain()
burnin= int(0.2*len(samples[:,0])) #Burnin del 20%
thin = 1
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$\Omega_{m}$','b'], 'AGN HS')
analisis.graficar_cadenas()
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False, color='k')
analisis.reportar_intervalos(discard=burnin, thin=thin)
