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
with np.load('valores_medios_EXP_CC+SN+BAO_4params.npz') as data:
    sol = data['sol']

#%% Importo las cadenas
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/EXP')
os.chdir(path_datos_global+'/Resultados_cadenas')
with np.load('sample_EXP_CC+SN+BAO_4params_deriv.npz') as data:
    ns = data['new_samples']
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'')
                    #'Supernovas tipo IA + Cronómetros Cósmicos + BAO')
analisis.graficar_cadenas_derivs()
analisis.graficar_contornos(sol, poster=True,color='r')
analisis.reportar_intervalos(sol)
