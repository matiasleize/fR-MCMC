"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np

import sys
import os
from os.path import join as osjoin


path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
path_datos_global = '/home/matias/Documents/Tesis/'

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador, magn_aparente_teorica
from funciones_cambio_parametros import params_fisicos_to_modelo

def chi_2_supernovas(muth,magn_aparente_obs,M_abs,C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''
    sn = len(muth)
    muobs =  magn_aparente_obs - M_abs
    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(C_invertida,deltamu)
    chi2 = np.dot(transp,aux)/sn
    return chi2

def params_to_chi2(cond_iniciales, theta, params_fijos, zcmb, zhel, Cinv, mb):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos de
    supernovas'''

    [Mabs,b,omega_m] = theta
    [H_0,n] = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)
    params_modelo = [c1,r_hs,c2,n] #de la cruz: [b,c(*H0^2),d,r_0,n]

    z,E = integrador(cond_iniciales, params_modelo)
    muth = magn_aparente_teorica(z,E,zhel,zcmb)
    chi = chi_2_supernovas(muth,mb,Mabs,Cinv)
    return chi

def params_to_chi2_H0(cond_iniciales, theta, params_fijos, z_data, H_data, dH):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas'''

    [omega_m,b,H_0] = theta
    n = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b)
    params_modelo = [c1,r_hs,c2,n] #de la cruz: [b,c(*H0^2),d,r_0,n]

    z,E = integrador(cond_iniciales, params_modelo)
    muth = magn_aparente_teorica(z,E,zhel,zcmb)
    chi = chi_2_supernovas(muth,mb,Mabs,Cinv)
    return chi
