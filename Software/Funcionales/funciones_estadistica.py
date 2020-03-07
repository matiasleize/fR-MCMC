"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np

'''En este file deberían in la definición de chi2 de las SN1a y
params_to_chi2 de SN1A.'''

def chi_2(muth,magn_aparente_obs,M_abs,C_invertida):
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
