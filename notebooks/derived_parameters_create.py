import numpy as np; #np.random.seed(42)
import emcee
from matplotlib import pyplot as plt
import yaml

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_global = os.path.dirname(path_git)

os.chdir(path_git); os.sys.path.append('./fr_mcmc/')
from utils.plotter import Plotter
from utils.derived_parameters import derived_parameters
from solve_sys import Hubble_th

from config import cfg as config

os.chdir(path_git + '/fr_mcmc/plotting/')

def parameters_labels(index, model):
    if model == 'LCDM':
        if index == 4:
            return ['$M_{abs}$', '$r_{d}$', r'$\\Omega_m$', '$H_{0}$']
        elif index == 31:
            return ['$M_{abs}$', r'$\\Omega_m$', '$H_{0}$']
        elif index == 32:
            return ['$r_{d}$', r'$\\Omega_m$', '$H_{0}$']
        elif index == 21:
            return [r'$\\Omega_m$', '$H_{0}$']

    elif (model == 'HS' or model == 'ST' or model == 'EXP'):
        if index == 5:
            return ['$M_{abs}$', '$r_{d}$', '$\\Omega_{m}^{LCDM}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 41:
            return ['$M_{abs}$', '$r_{d}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 42:
            return ['$M_{abs}$', '$\\Omega_{m}^{LCDM}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 43:
            return ['$r_{d}$', '$\\Omega_{m}^{LCDM}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 31:
            return ['$M_{abs}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 32:
            return ['$r_{d}$', r'$b$', '$H_{0}^{LCDM}$']
        elif index == 33:
            return [r'$\\Omega_m$^{LCDM}', r'$b$', '$H_{0}^{LCDM}$']



#@jit
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

model = 'HS'
output_dir = '/fR-output/HS/'
index = 42
#filename='sample_HS_PPS_BAO_full_4params'
#filename='sample_HS_BAO_full_3params'
filename='sample_HS_PPS_4params'

output_path = path_global + output_dir + filename
os.chdir(output_path)

parameters_label = parameters_labels(index, model)

reader = emcee.backends.HDFBackend(filename + '.h5')
samples = reader.get_chain()

burnin = int(len(samples[:,0])-1000/12); thin=1
print(len(samples[:,0]))
print(len(samples[:,0])-1000)
#burnin = int(0.999*len(samples[:,0])); thin=1

sampler=reader
flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
len_chain=flat_samples.shape[0]
print(len_chain)
new_samples = np.full_like(flat_samples,1)


#@jit
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
#print(new_samples)
np.savez(filename+'_deriv', new_samples=new_samples)
