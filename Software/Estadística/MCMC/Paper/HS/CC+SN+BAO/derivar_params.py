import numpy as np
import emcee
import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git+'/Software/Funcionales')
from funciones_parametros_derivados import parametros_derivados

#Rellenar acá:
model = 'HS'
datasets = 'CC+SN+BAO'
num_params = '4params'
root_directory = path_datos_global+'/Resultados_cadenas/Paper/'+model
#root_directory=path_datos_global+'/Resultados_cadenas'
root_directory
os.chdir(root_directory)
filename = 'sample_'+model+'_'+datasets+'_'+num_params
filename_h5 = filename+'.h5'
reader = emcee.backends.HDFBackend(filename_h5)
nwalkers, ndim = reader.shape #Numero de caminantes y de parametros

#%%%
samples = reader.get_chain()
burnin= int(0.2*len(samples[:,0])) #Burnin del 20%
thin = 1

#%% Defino el burnin y el thin a partir de tau o los pongo a mano
tau = reader.get_autocorr_time()
#burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
#%%
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
print(len(samples)) #numero de pasos efectivos
print('Tiempo estimado:{} min'.format(len(samples)/60))
new_samples = parametros_derivados(reader,discard=burnin,thin=thin,model=model)

#%%
np.savez(filename+'_deriv', new_samples=new_samples)
#dir = path_datos_global+'/Resultados_cadenas/posprocesado'
os.chdir(root_directory)
with np.load(filename+'_deriv.npz') as data:
    ns = data['new_samples']
