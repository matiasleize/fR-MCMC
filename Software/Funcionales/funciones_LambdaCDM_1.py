"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from numpy.linalg import inv

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg
#%%

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

def magn_aparente_teorica(z, H, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(H**(-1), z, initial=0)
    dc_int = interp1d(z, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor

    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi_2_supernovas(muth, muobs, C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''

    deltamu = muth - muobs #vector fila
    transp = np.transpose(deltamu) #vector columna
    aux = np.dot(C_invertida,transp) #vector columna
    chi2 = np.dot(deltamu,aux) #escalar

    #chi2 = np.dot(np.dot(transp,C_invertida),deltamu) #cuenta de Caro (da igual)
    return chi2


def params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv, mb,
                    cantidad_zs=int(10**6), Mfijo=False, omega_solo=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    omega_m = theta
    [Mabs, H_0] = params_fijos

    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)

    muth = magn_aparente_teorica(z, H, zcmb, zhel)
    muobs =  mb - Mabs

    chi = chi_2_supernovas(muth, muobs, Cinv)
    chi_norm = chi / (len(zcmb) - 1)
    return chi_norm
