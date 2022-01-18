"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
from numba import jit
import numpy as np

import sys
import os
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int_sist_1 import Hubble_teorico_1


@jit
def parametros_derivados(sampler,discard, thin,model='EXP'):
	'''Esta función convierte las cadenas de omega_m y H0 de LCDM
	en las cadenas de omega_m y H0 fisicas'''
	flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	len_chain=flat_samples.shape[0]
	new_samples = np.full_like(flat_samples,1)
	for i in range(len_chain):
	#for i in range(1000,1100):
		if len(flat_samples[0,:])==3:
			omega_m_lcdm = flat_samples[i,0]
			b = flat_samples[i,1]
			H0_lcdm = flat_samples[i,2]
			_, Hubble = Hubble_teorico([omega_m_lcdm,b,H0_lcdm], verbose=False, model=model)
			H0 = Hubble[0]
			omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

			new_samples[i,0] = omega_m
			new_samples[i,1] = b
			new_samples[i,2] = H0
		elif len(flat_samples[0,:])==4:
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

	#	print('Completado: {}/{}'.format(i,len_chain))
	return new_samples
