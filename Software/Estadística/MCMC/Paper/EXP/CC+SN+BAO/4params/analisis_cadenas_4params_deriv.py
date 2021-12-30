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
sys.path.append('./Software/Funcionales')
from hdi import hpd_grid

#%% Importo las cadenas
#os.chdir(path_datos_global+'/Resultados_cadenas/Paper/12 cadenas/EXP')
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/EXP')
with np.load('sample_EXP_CC+SN+BAO_4params_deriv.npz') as data:
    ns = data['new_samples']

#%%

intervals,_,_,_ = hpd_grid(ns[:,2],alpha=0.32)
print(intervals)

intervals,_,_,_ = hpd_grid(ns[:,2],alpha=0.05)
print(intervals)
#Te hago una consulta, recién estuve intantando conectarme al backdoor y me dice que la contraseño es incorrecta. A vos también te pasa? Es raro porque ayer pude conectarme lo más bien
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'EXP (CC+SN+BAO)')
#]analisis.graficar_cadenas_derivs()
analisis.graficar_contornos(discard=0,thin=1,poster=False,color='r')
analisis.reportar_intervalos(discard=0,thin=1)
