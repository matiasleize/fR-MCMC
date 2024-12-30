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
from ML import H_ML, ML_limits

from config import cfg as config


model = 'HS'
output_dir = '/fR-output/HS/'
index = 42
#filename='sample_HS_PPS_BAO_full_4params'
#filename='sample_HS_BAO_full_3params'
filename='sample_HS_PPS_4params'

output_path = path_global + output_dir + filename
os.chdir(output_path)

reader = emcee.backends.HDFBackend(filename + '.h5')
samples = reader.get_chain()


#burnin = int(0.999*len(samples[:,0])); thin=1
#burnin = int(len(samples[:,0])-100/12); thin=1
burnin = int(0.2*len(samples[:,0])); thin=1
print(len(samples[:,0]))

sampler=reader
flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
len_chain=flat_samples.shape[0]
print(len_chain)
new_samples = np.full_like(flat_samples,1)

if len(flat_samples[0,:])==3:

    print(min(flat_samples[:,0]),max(flat_samples[:,0]))
    print(min(flat_samples[:,2]),max(flat_samples[:,2]))

    #@jit
    omega_m_lcdm = flat_samples[:,0]
    b = flat_samples[:,1]
    H0_lcdm = flat_samples[:,2]
    theta_example = [b, omega_m_lcdm, H0_lcdm, flat_samples[:,0]]
    H0 = H_ML(0, theta_example, post=True)
    omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

    new_samples[:,0] = omega_m
    new_samples[:,1] = b
    new_samples[:,2] = H0

    print(new_samples)

    #Omega_m
    #plt.hist(flat_samples[:,0],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.hist(new_samples[:,0],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.show()

    #H0
    #plt.hist(flat_samples[:,2],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.hist(new_samples[:,2],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.show()

if len(flat_samples[0,:])==4:
    print(min(flat_samples[:,1]),max(flat_samples[:,1]))
    print(min(flat_samples[:,3]),max(flat_samples[:,3]))

    #@jit
    omega_m_lcdm = flat_samples[:,1]
    b = flat_samples[:,2]
    H0_lcdm = flat_samples[:,3]
    theta_example = [b, omega_m_lcdm, H0_lcdm, flat_samples[:,0]]
    H0 = H_ML(0, theta_example, post=True)
    omega_m  = omega_m_lcdm * (H0_lcdm/H0)**2

    new_samples[:,0] = flat_samples[:,0]
    new_samples[:,1] = omega_m
    new_samples[:,2] = b
    new_samples[:,3] = H0

    print(new_samples)

    #Omega_m
    #plt.hist(flat_samples[:,1],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.hist(new_samples[:,1],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.show()

    #H0
    #plt.hist(flat_samples[:,3],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.hist(new_samples[:,3],bins=int(np.sqrt(len(new_samples[:,0]))))
    #plt.show()
#print(new_samples)
np.savez(filename+'_ML_deriv', new_samples=new_samples)