"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos

def params_fisicos_to_modelo(omega_m, b, H_0, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b y H_0)
    y devuelve los parametros del modelo c1, c2 y R_HS'''
    h0 = H_0 / 100
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = (H_0/c_luz_norm)**2 * 6 * (1 - omega_m)  / (alpha * (omega_m * beta * h0 ** 2))
    c_2 =  (2/(aux * b)) ** n
    c_1 =  aux * c_2
    r_hs = H_0**2 * 6 * (1-omega_m) /aux
    return c_1, r_hs, c_2


#%%
#H_0=73.48
#b_true = 2
#omega_m_true = 0.24
#c1,r_hs,c2 = params_fisicos_to_modelo(omega_m_true,b_true)
#print(c1,r_hs/H_0**2,c2)
#c1_true = 1
#r_hs_true = 0.24
#c2_true = 1/19
#%%
