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
#os.chdir(path_datos_global+'/Resultados_cadenas')
with np.load('sample_EXP_CC+SN+BAO+AGN_4params_deriv.npz') as data:
    ns = data['new_samples']
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'EXP (CC+SN+BAO+AGN)')
analisis.graficar_cadenas_derivs()
analisis.graficar_contornos(poster=False,color='r')
plt.savefig('/home/matias/Desktop/Entrega 17_09/Corridas/CC+SN+BAO+AGN')
analisis.reportar_intervalos(discard=1,thin=1)
