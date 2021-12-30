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
with np.load('sample_EXP_CC+SN_4params_deriv.npz') as data:
    ns = data['new_samples']
#%%
# Saving the array in a text file
#np.savez('/home/matias/Desktop/EXP_CC+SN_bs.npz', bs=ns[:,2])
#with np.load('/home/matias/Desktop/EXP_CC+SN_bs.npz') as data:
#    bs = data['bs']

len(ns)
b = ns[:,2]
b.shape
c = np.histogram(b, bins=100)[0]
d = np.histogram(b, bins=100)[1]
c_sum = np.cumsum(c)
c
c.max()
index_max = np.where(c==c.max())[0][0]
index_max

0.68*44040
index = np.where(c_sum==30095)[0]

d[index]
#%%
intervals,_,_,_ = hpd_grid(ns[:,2],alpha=0.32)
intervals

intervals,_,_,_ = hpd_grid(ns[:,2],alpha=0.05)
intervals
#%%
%matplotlib qt5
analisis = Graficador(ns, ['$M_{abs}$','$\Omega_{m}$','b','$H_{0}$'],'CC+SnIA (EXP) - Params Deriv')
analisis.graficar_cadenas_derivs()
analisis.graficar_contornos(discard=0,thin=1,poster=False,color='r')
plt.savefig('/home/matias/Desktop/CC+SN')
analisis.reportar_intervalos(discard=0,thin=1)
