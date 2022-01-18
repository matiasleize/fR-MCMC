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
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/12 cadenas/HS')
#os.chdir(path_datos_global+'/Resultados_cadenas')
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/HS')

with np.load('sample_HS_CC+SN+BAO+AGN_4params_deriv.npz') as data:
    ns = data['new_samples']
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'HS (SnIA + CC + BAO+AGN)')
analisis.graficar_contornos(discard=0, thin=1,poster=False,color='r')
plt.savefig('/home/matias/Desktop/2.png')
analisis.reportar_intervalos(discard=0,thin=1)