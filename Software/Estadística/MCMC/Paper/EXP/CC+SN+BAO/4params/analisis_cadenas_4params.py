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
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/12 cadenas/EXP')
filename = "sample_EXP_CC+SN+BAO_4params.h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
#%%
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'')
                    #'Supernovas tipo IA + Cronómetros Cósmicos + BAO')
analisis.graficar_contornos(discard=burnin, thin=thin,
                            poster=False,color='r')
analisis.graficar_cadenas()
analisis.reportar_intervalos()
#%%
%matplotlib qt5
bs = reader.get_chain(discard=burnin,thin=thin,flat=True)[:,2]
max(bs)
betas = 2/bs
#betas = betas[betas<50]
len(betas)
bi=int(np.sqrt(len(betas)))
plt.close()
plt.figure()
plt.grid()
plt.hist(betas,bins=bi,density=True);
