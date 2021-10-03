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
os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones')
with np.load('valores_medios_LCDM_CC+SN_4params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas')
terminacion=""
filename = "sample_LCDM_CC+SN_4params"+terminacion+".h5"
reader = emcee.backends.HDFBackend(filename)

# Algunos valores
tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))

#sample = reader.get_chain()
#burnin = int(0.2*len(sample[:,0]))

#%%
#burnin=100
#thin=5
%matplotlib qt5
analisis = Graficador(reader, ['$M_{abs}$','$\Omega_{m}$','$H_{0}$'],'CC+SN (LCDM)')
                    #'Supernovas tipo IA + Cronómetros Cósmicos + BAO')
analisis.graficar_contornos(discard=burnin, thin=thin, poster=False,color='r')
plt.savefig('/home/matias/Desktop/Moves/contornos_{}'.format(terminacion))
#%%
analisis.graficar_cadenas(num_chains=5)
plt.savefig('/home/matias/Desktop/Moves/cadena_{}'.format(terminacion))
analisis.reportar_intervalos(discard=burnin, thin=thin)
