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
#%%

%matplotlib qt5

os.chdir(path_datos_global+'/Resultados_cadenas/')
filename_mati = "sample_HS_CC+SN_4params_augusto.h5"
reader_mati = emcee.backends.HDFBackend(filename_mati)
aux = reader_mati.get_chain()
burnin= burnin=int(0.2*len(aux[:,0]))
thin=1
#tau = reader_mati.get_autocorr_time()
#thin = int(0.5 * np.min(tau))
samples_mati_20 = reader_mati.get_chain(flat=True,discard=burnin,thin=thin)[:,2]


os.chdir(path_datos_global+'/Resultados_cadenas')
filename = 'CC+SnIa_HS_NN.h5' #Corridas de Augusto
reader_agus = emcee.backends.HDFBackend(filename)
samples = reader_agus.get_chain()
burnin= burnin=int(0.2*len(samples[:,0]))
thin=1
#tau = reader_agus.get_autocorr_time()
#thin = int(0.5 * np.min(tau))
samples_agus = reader_agus.get_chain(flat=True,discard=burnin,thin=thin)[:,0]


os.chdir(path_datos_global+'/Resultados_cadenas/')
filename_mati_1 = "sample_HS_CC+SN_4params.h5"
reader_mati_1 = emcee.backends.HDFBackend(filename_mati_1)
aux = reader_mati_1.get_chain()
burnin= burnin=int(0.2*len(aux[:,0]))
thin=1
#tau = reader_mati_1.get_autocorr_time()
#thin = int(0.5 * np.min(tau))
samples_mati_12 = reader_mati_1.get_chain(flat=True,discard=burnin,thin=thin)[:,2]


#%%
plt.close()
plt.figure()
plt.title('CC+Sn')
plt.grid()
plt.hist(samples_mati_12,alpha=0.5,label='Matías 12 cadenas',density=True,bins='auto')
plt.hist(samples_agus,alpha=0.5,label='Augusto 20 cadenas',density=True,bins='auto')
plt.hist(samples_mati_20,alpha=0.5,label='Matías 20 cadenas',density=True,bins='auto')
plt.xlabel('b')
plt.legend()
plt.savefig('/home/matias/Desktop/b')

#%% Distinto bineado
bineado = lambda x: np.int(np.sqrt(len(x))) #Otra opcion
plt.close()
plt.figure()
plt.title('CC+Sn')
plt.grid()
plt.hist(samples_mati_1,alpha=0.5,label='Matías 12 cadenas',density=True,bins=bineado(samples_mati_1))
plt.hist(samples_agus,alpha=0.5,label='Augusto 20 cadenas',density=True,bins=bineado(samples_agus))
plt.hist(samples_mati,alpha=0.5,label='Matías 20 cadenas',density=True,bins=bineado(samples_mati))
plt.xlabel('b')
plt.legend()
plt.savefig('/home/matias/Desktop/hist_3')
#%%
#from scipy.stats import wilcoxon

#a=np.random.normal(loc=0.1,scale=2,size=10000)
#b=np.random.normal(scale=2,size=10000)
#wilcoxon(a,b)

#wilcoxon(samples_mati_12,samples_mati_20[:len(samples_mati_12)])
#wilcoxon(samples_agus[:len(samples_mati_20)],samples_mati_20)
#wilcoxon(samples_mati_12,samples_agus[:len(samples_mati_12)])
len(samples_agus)
len(samples_mati_20)
len(samples_mati_12)
