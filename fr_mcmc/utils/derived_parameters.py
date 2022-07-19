"""
Calculate the derivate parameters. IMPORTANT: It Doesn't work for all the indeces yet.
"""

from numba import jit
import numpy as np

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from solve_sys import Hubble_th


@jit
def derived_parameters(sampler,discard, thin,model='EXP'):
	'''Convert LCDM chains into physical chains (for Omega_m and H_0 parameters).'''
	flat_samples = sampler.get_chain(discard=discard, flat=True, thin=thin)
	len_chain=flat_samples.shape[0]
	new_samples = np.full_like(flat_samples,1)
	for i in range(len_chain):
		if len(flat_samples[0,:])==3:
			omega_m_lcdm = flat_samples[i,0]
			b = flat_samples[i,1]
			H0_lcdm = flat_samples[i,2]
			_, Hubble = Hubble_th([omega_m_lcdm,b,H0_lcdm], verbose=False, model=model)
			H0 = Hubble[0]
			omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

			new_samples[i,0] = omega_m
			new_samples[i,1] = b
			new_samples[i,2] = H0
		elif len(flat_samples[0,:])==4:
			omega_m_lcdm = flat_samples[i,1]
			b = flat_samples[i,2]
			H0_lcdm = flat_samples[i,3]
			_, Hubble = Hubble_th([omega_m_lcdm,b,H0_lcdm], verbose=False, model=model)
			H0 = Hubble[0]
			omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

			new_samples[i,0] = flat_samples[i,0]
			new_samples[i,1] = omega_m
			new_samples[i,2] = b
			new_samples[i,3] = H0

		#print('Complete: {}/{}'.format(i,len_chain))
	return new_samples
