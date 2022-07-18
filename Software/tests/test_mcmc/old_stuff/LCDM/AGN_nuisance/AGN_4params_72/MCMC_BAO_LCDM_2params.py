import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import emcee
import corner
from scipy.interpolate import interp1d
import time
np.random.seed(1)

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from sampling import MCMC_sampler
from data import leer_data_AGN
from AGN import params_to_chi2_AGN_nuisance

beta_true =  7.735
gamma_true = 0.648
delta_true = 0.235
#omega_m = 0.5
H0_true =  72


os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN/')
data_agn = leer_data_AGN('table3.dat')

params_fijos = [beta_true, gamma_true, delta_true, H0_true]

os.chdir(path_git+'/Software/Estadística/Resultados_simulaciones/LCDM')
with np.load('valores_medios_LCDM_AGN_4params_nuisance.npz') as data:
    sol = data['sol']
sol = 0.6

log_likelihood = lambda theta: -0.5 * params_to_chi2_AGN_nuisance(theta,params_fijos, data_agn,model='LCDM')

#%% Definimos las funciones de prior y el posterior
def log_prior(theta):
    omega_m = theta
    if (0.5 < omega_m < 0.99):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


pos = sol * (1 +  0.01 * np.random.randn(20,1))
MCMC_sampler(log_probability,pos,
            filename = "sample_LCDM_AGN_4params_nuisance_72.h5",
            witness_file = 'witness_67.txt',
            witness_freq = 10,
            max_samples = 200000)
