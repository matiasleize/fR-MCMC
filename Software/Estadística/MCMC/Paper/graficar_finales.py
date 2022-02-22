#%%
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

#LCDM
os.chdir(path_datos_global+'/Resultados_cadenas/Paper_v1/LCDM')
filename_1 = "sample_LCDM_CC+SN+BAO+AGN_3params.h5"
reader_1 = emcee.backends.HDFBackend(filename_1)
samples_1 = reader_1.get_chain()
burnin_1 = int(0.2*len(samples_1[:,0])) #Burnin del 20%
samples_1 = reader_1.get_chain(flat=True, discard=burnin_1,thin=1)

#Hu-Sawicki
os.chdir(path_datos_global+'/Resultados_cadenas/Paper_v1/HS')
with np.load('sample_HS_CC+SN_4params_deriv.npz') as data:
    samples_2 = data['new_samples']

#Exponencial
os.chdir(path_datos_global+'/Resultados_cadenas/Paper_v1/EXP')
with np.load('sample_EXP_CC+SN_4params_deriv.npz') as data:
    samples_3 = data['new_samples']


#%%
#%matplotlib qt5
ndim = 2
names = ['\Omega_{m}', 'H_{0}']
labels=names


aux_2=np.delete(samples_2,0,1)
samples_2=np.delete(aux_2,1,1)
aux_3=np.delete(samples_3,0,1)
samples_3=np.delete(aux_3,1,1)



samples1 = MCSamples(samples=samples_1[:,1:], names = names, labels = labels)
samples1 = samples1.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

samples2 = MCSamples(samples=samples_2, names = names, labels = labels)
samples2 = samples2.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

samples3 = MCSamples(samples=samples_3, names = names, labels = labels)
samples3 = samples3.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})


#%%
g = plots.get_subplot_plotter()
g.triangle_plot([samples2, samples3,samples1],
                contour_colors=['r','b','g'],
                filled=True, params = names ,
                contour_lws=1.5,
#                param_limits = dict
                legend_labels = ['Hu-Sawicki','Exponential','LCDM'])
plt.savefig('/home/matias/Desktop/contornos_1.png')

#%%
g = plots.get_subplot_plotter()
g.triangle_plot([samples2,samples1],
                contour_colors=['r','g'],
                filled=True, params = names ,
                contour_lws=1.5,
#                param_limits = dict
                legend_labels = ['Hu-Sawicki','LCDM'])
plt.savefig('/home/matias/Desktop/contornos_12.png')
# %%
g = plots.get_subplot_plotter()
g.triangle_plot([samples3,samples1],
                contour_colors=['b','g'],
                filled=True, params = names ,
                contour_lws=1.5,
#                param_limits = dict
                legend_labels = ['Exponential','LCDM'])
plt.savefig('/home/matias/Desktop/contornos_13.png')
