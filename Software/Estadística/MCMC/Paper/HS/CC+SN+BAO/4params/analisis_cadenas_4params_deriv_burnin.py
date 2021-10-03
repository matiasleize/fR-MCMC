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
#os.chdir(path_datos_global+'/Resultados_cadenas')
with np.load('sample_HS_CC+SN+BAO_4params_deriv.npz') as data:
    ns = data['new_samples']

burnin=int(0.2*len(ns[:,0]))
burnin
ns = ns[burnin:,:]
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'HS (SnIA + CC + BAO)')
analisis.graficar_cadenas_derivs()
analisis.graficar_contornos(poster=False,color='r')
#plt.savefig('/home/matias/Desktop/Entrega 17_09/Corridas/CC+SN+BAO')
analisis.reportar_intervalos(hdi=True)
