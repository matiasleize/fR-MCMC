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
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/EXP')
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/12 cadenas/EXP')
filename = "sample_EXP_CC+SN+BAO+AGN_4params.h5"
#filename = "sample_EXP_CC+SN+BAO+AGN_4params_2.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores

samples = reader.get_chain()
burnin= burnin=int(0.2*len(samples[:,0]))
thin=1

#Numer de pasos totales
samples = reader.get_chain(flat=True,thin=1, discard=burnin)
len(samples[:])

#%%
#thin = 2
#burnin = 200
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}^{\Lambda CDM}$','b','$H_{0}^{\Lambda CDM}$'],'SNIA + CC + BAO + AGN(EXP)')
analisis.graficar_contornos(discard=burnin, thin=1,poster=False,color='r')
plt.savefig('/home/matias/Desktop/20 cadenas.png')
#analisis.graficar_cadenas()
analisis.reportar_intervalos(discard=burnin, thin=thin)
