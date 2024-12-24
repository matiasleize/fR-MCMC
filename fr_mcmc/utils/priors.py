"""
Definition of the log prior distribution in terms 
of the parameters of the model. 
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000

import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir

os.chdir(path_git); os.sys.path.append('./fr_mcmc/utils/')
from config import cfg as config

model = config.MODEL
[M_min, M_max] = config.M_PRIOR
[omega_m_min, omega_m_max] = config.OMEGA_M_PRIOR
if model != 'LCDM':
    [b_min, b_max] = config.B_PRIOR
[H0_min, H0_max] = config.H0_PRIOR


# TODO: Implement priors for HS! index 4 most
def log_prior(theta, index):
    '''
    Define the logarithm of the prior distribution
    '''
    if index == 4:
        M, omega_m, b, H0 = theta
        if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
            return 0.0
    elif index == 31:
        omega_m, b, H0 = theta
        if (omega_m_min < omega_m < omega_m_max and b_min < b < b_max and H0_min < H0 < H0_max):
            return 0.0
    elif index == 32:
        if config.OMEGA_M_ASTRO_PRIOR == True: #Omega_m gaussian prior
            M, omega_m, H0 = theta
            if not M_min < M < M_max and H0_min < H0 < H0_max:
                return -np.inf
            mu = 0.19
            sigma = 0.06
            return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(omega_m-mu)**2/sigma**2  
        elif config.M_ABS_CM_PRIOR == True: #M_abs Camarena & Marra prior
            M, omega_m, H0 = theta
            if not omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max:
                return -np.inf
            mu = -19.2435
            sigma = 0.0373
            return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(M-mu)**2/sigma**2  
        else: #Flat prior
            M, omega_m, H0 = theta
            if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
                return 0.0
    elif index == 33:
        M, omega_m, b = theta
        if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max and b_min < b < b_max):
            return 0.0
    elif index == 21:
        omega_m, b = theta
        if (omega_m_min < omega_m < omega_m_max and b_min < b < b_max):
            return 0.0
    elif index == 22:
        omega_m, H0 = theta
        if (omega_m_min < omega_m < omega_m_max and H0_min < H0 < H0_max):
            return 0.0
    elif index == 23:
        M, omega_m = theta
        if (M_min < M < M_max and omega_m_min < omega_m < omega_m_max):
            return 0.0
    elif index == 1:
        omega_m = theta
        if omega_m_min < omega_m < omega_m_max:
            return 0.0
    return -np.inf