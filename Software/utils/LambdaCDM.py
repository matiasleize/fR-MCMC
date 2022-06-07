"""
Functions related to LCDM model
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
#import camb #No lo reconoce la compu del df
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
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
    Calculation of the Hubble parameter.
    Here we neclect the radiation (it holds 
    that \Omega_r + \Omega_m + \Omega_L = 1).
    '''
    H = H_0 * E_LCDM(z, omega_m)
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
#%%

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #Despues poner esto en un lugar pertinente.
    #N = len(data_agn[0]) #Número de datos
    #P = 1 #Número de parámetros
    #np.sqrt(2*(N-P))/(N-P) #Fórmula del error en el chi2 reducido
