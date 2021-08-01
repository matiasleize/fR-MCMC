"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import time
import emcee
import corner
import seaborn as sns
import pandas as pd
import sys
import os

from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import Hubble_teorico
def parametros_derivados(sampler,discard, thin,model='EXP'):
	'''Esta función convierte las cadenas de omega_m y H0 de LCDM
	en las cadenas de omega_m y H0 fisicas'''
	flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	len_chain=flat_samples.shape[0]
	new_samples = np.full_like(flat_samples,1)
	#for i in range(len_chain):
	for i in range(20):
		omega_m_lcdm = flat_samples[i,1]
		b = flat_samples[i,2]
		H0_lcdm = flat_samples[i,3]
		_, Hubble = Hubble_teorico([omega_m_lcdm,b,H0_lcdm], verbose=False, model=model)
		H0 = Hubble[0]
		omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

		new_samples[i,0] = flat_samples[i,0]
		new_samples[i,1] = omega_m
		new_samples[i,2] = b
		new_samples[i,3] = H0

		print('Completado: {}/{}'.format(i,len_chain))
	return new_samples


#%%
if __name__ == '__main__':
	import numpy as np
	import emcee
	import seaborn as sns
	from matplotlib import pyplot as plt
	import corner



	#%%
	os.chdir(path_datos_global+'/Resultados_cadenas/Paper')
	filename = "sample_EXP_CC+SN+AGN_4params.h5"
	reader = emcee.backends.HDFBackend(filename)
	# Algunos valores
	nwalkers, ndim = reader.shape
	tau = reader.get_autocorr_time()
	burnin = int(2 * np.max(tau))
	thin = int(0.5 * np.min(tau))
	new_sample = parametros_derivados(reader,discard=burnin,thin=thin,model='EXP')
	samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
	print(nwalkers,ndim)
	print(samples.shape[0]) #numero de pasos
	print(tau)
