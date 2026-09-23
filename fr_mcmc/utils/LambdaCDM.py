"""
Functions related to LCDM model
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #meters/seconds
c_luz_km = c_luz/1000
from constants import Omega_r
#%%

def E_LCDM(z, omega_m):
    '''
    Calculation of the normalized Hubble parameter, independent
    of the Hubble constant H0.
    '''
    omega_lambda = 1 - omega_m
    E = np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return E

def H_LCDM(z, omega_m, H_0):
    '''
    Calculation of the Hubble parameter (flat, \Omega_r + \Omega_m + \Omega_L = 1).
    The radiation (photons + massless neutrinos) is included if constants.RADIATION
    is True, with \Omega_r = constants.OMEGA_R_H2 / h^2.
    '''
    omega_r = Omega_r(H_0)
    H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + 1 - omega_m - omega_r)
    return H

def H_LCDM_rad(z, omega_m, H_0):
    '''
    Calculation of the Hubble parameter. Here it holds that
    \Omega_r + \Omega_m + \Omega_L = 1 
    '''
    omega_r = 4.18343*10**(-5) / (H_0/100)**2
    omega_lambda = 1 - omega_m - omega_r

    if isinstance(z, (np.ndarray, list)):
        H = H_0 * np.sqrt(omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)
    else:
        H = H_0 * (omega_r * (1 + z)**4 + omega_m * (1 + z)**3 + omega_lambda)**(1/2)

    return H