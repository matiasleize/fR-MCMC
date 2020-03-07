"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
import sys
import os
from os.path import join as osjoin
from scipy.interpolate import interp1d

from scipy.constants import c as c_luz #metros/segundos
path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo

def chi_2_cronometros(H_data, H_teo, dH):
    chi2 = np.sum((H_data-H_teo)**2/dH**2)
    return chi2

def params_to_chi2(cond_iniciales, theta, params_fijos, z_data, H_data, dH):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [omega_m,b] = theta
    [H_0,n] = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)
    params_modelo = [c1,r_hs,c2,n] #de la cruz: [b,c(*H0^2),d,r_0,n]
    z,E = integrador(cond_iniciales, params_modelo)
    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi

def params_to_chi2_H0(cond_iniciales, theta, params_fijos, z_data, H_data, dH):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [omega_m,b,H_0] = theta
    n = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)
    params_modelo = [c1,r_hs,c2,n] #de la cruz: [b,c(*H0^2),d,r_0,n]
    z,E = integrador(cond_iniciales, params_modelo)
    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi
