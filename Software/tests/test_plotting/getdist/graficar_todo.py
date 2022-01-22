import numpy as np
from getdist import plots, MCSamples
from matplotlib import pyplot as plt
import emcee
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/utils/Clases')
from graficador import Graficador

#%% Importo las cadenas
os.chdir(path_datos_global+'/Resultados_cadenas/Paper/LCDM')

filename_1 = "sample_LCDM_CC+SN_4params.h5"
reader_1 = emcee.backends.HDFBackend(filename_1)

filename_2 = "sample_LCDM_CC+SN+AGN_3params.h5"
reader_2 = emcee.backends.HDFBackend(filename_2)

filename_3 = "sample_LCDM_CC+SN+BAO_3params.h5"
reader_3 = emcee.backends.HDFBackend(filename_3)

filename_4 = "sample_LCDM_CC+SN+BAO+AGN_3params.h5"
reader_4 = emcee.backends.HDFBackend(filename_4)


# Algunos valores
samples_1 = reader_1.get_chain()
burnin_1 = int(0.2*len(samples_1[:,0])) #Burnin del 20%

samples_2 = reader_2.get_chain()
burnin_2 = int(0.2*len(samples_2[:,0])) #Burnin del 20%

samples_3 = reader_3.get_chain()
burnin_3 = int(0.2*len(samples_3[:,0])) #Burnin del 20%

samples_4 = reader_4.get_chain()
burnin_4 = int(0.2*len(samples_4[:,0])) #Burnin del 20%


thin = 1
#%%
samples_1 = reader_1.get_chain(flat=True, discard=burnin_1,thin=1)
samples_2 = reader_2.get_chain(flat=True, discard=burnin_2,thin=1)
samples_3 = reader_3.get_chain(flat=True, discard=burnin_3,thin=1)
samples_4 = reader_4.get_chain(flat=True, discard=burnin_4,thin=1)
#%%
%matplotlib qt5
ndim = 3
names = ['M_{abs}','\Omega_{m}', 'H_{0}']
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

samples4 = MCSamples(samples=samples_4,names = names, labels = labels)
samples4 = samples4.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
             settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3, 'smooth_scale_1D':0.3})

g = plots.get_subplot_plotter()
g.triangle_plot([samples1, samples2, samples3, samples4],
                filled=True, params = names,
                contour_colors=['grey','g','r','b'],
                contour_lws=1,
                legend_labels = ['CC+SN', 'CC+SN+AGN', 'CC+SN+BAO','CC+SN+BAO+AGN'])

plt.savefig('/home/matias/Desktop/contornos_LCDM.png')


#%% Importo las cadenas
def graficar_contornos(self, discard=0, thin=1, color='b'):
    
    if model == LCDM:
        filename_1 = "sample_LCDM_CC+SN_4params.h5"
        reader_1 = emcee.backends.HDFBackend(filename_1)
        samples_1 = reader_1.get_chain()
        samples = reader_1.get_chain(flat=True, discard=discard, thin=thin)

    else:
        with np.load('sample_HS_CC+SN_4params_deriv.npz') as data:
            samples = data['new_samples']

    #ndim = 3
    #names = ['M_{abs}','\Omega_{m}', 'H_{0}']
    #labels = names
    ndim = len(self.labels)
    samples1 = MCSamples(samples=self.samples, names=self.labels, labels=self.labels)
    samples1 = samples1.copy(label=r'Lowest-order with $0.3\sigma$ smoothing',
                settings={'mult_bias_correction_order':0,'smooth_scale_2D':0.3,
                'smooth_scale_1D':0.3})

    g = plots.get_subplot_plotter()
    g.triangle_plot(samples1,
                    filled=True, params=self.labels,
                    contour_colors=color,
                    contour_lws=1,
                    legend_labels='CC+SN+AGN')

