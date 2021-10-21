import numpy as np
from getdist import plots, MCSamples
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
with np.load('sample_EXP_CC+SN+AGN_4params_deriv.npz') as data:
    samples_1 = data['new_samples']

with np.load('sample_EXP_CC+SN+BAO_4params_deriv.npz') as data:
    samples_2 = data['new_samples']

with np.load('sample_EXP_CC+SN+BAO+AGN_4params_deriv.npz') as data:
    samples_3 = data['new_samples']

os.chdir(path_datos_global+'/Resultados_cadenas')
with np.load('sample_EXP_CC+SN_4params_deriv.npz') as data:
    samples_0 = data['new_samples']


#%%
%matplotlib qt5
ndim = 4
names = ['M_{abs}','\Omega_{m}','b','H_{0}']
labels=names
samples1 = MCSamples(samples=samples_1,names = names, labels = labels)
samples1 = samples1.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

samples2 = MCSamples(samples=samples_2,names = names, labels = labels)
samples2 = samples2.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

samples3 = MCSamples(samples=samples_3,names = names, labels = labels)
samples3 = samples3.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})
samples0 = MCSamples(samples=samples_0,names = names, labels = labels)
samples0 = samples0.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

g = plots.get_subplot_plotter()
g.triangle_plot([samples0,samples1, samples2, samples3], filled=True,legend_labels = ['CC+SN','CC+SN+AGN', 'CC+SN+BAO','CC+SN+BAO+AGN'])
#g.triangle_plot([samples0,samples1, samples2, samples3], filled=True, params = names ,legend_labels = ['CC+SN','CC+SN+AGN', 'CC+SN+BAO','CC+SN+BAO+AGN'])
plt.savefig('/home/matias/Desktop/contornos_EXP.png')
