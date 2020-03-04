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


def leer_data_cronometros(archivo):

    '''Toma la data de Pantheon y extrae la data de los redshifts zcmb y zhel
    su error dz, además de los datos de la magnitud aparente con su error:
    mb y dm. Con los errores de la magnitud aparente construye la
    matriz de correlación asociada. La función devuelve la información
    de los redshifts, la magnitud aparente y la matriz de correlación
    inversa.'''

    # leo la tabla de datos:
    z, h ,dh =np.loadtxt(archivo, usecols=(1,2,3),unpack=True)

    return z,h


def chi_2_cronometros(H_data, H_teo, dH):
    chi2 = np.sum((H_data-H_teo)**2/dH**2)
    return chi2

def params_fisicos_to_modelo(omega_m, b, H_0=73.48, n=1):
    '''Toma los parametros fisicos (omega_m y el parametro de distorsión b)
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

#%%
H_0=73.48
b_true = 2
omega_m_true = 0.24
c1,r_hs,c2 = params_fisicos_to_modelo(omega_m_true,b_true)
print(c1,r_hs/H_0**2,c2)
#c1_true = 1
#r_hs_true = 0.24
#c2_true = 1/19
#%%
