"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
import time
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000
#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m, b, H_0, n


def params_fisicos_to_modelo(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2'''
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = ((100/c_luz_norm)**2 * 6 * (1 - omega_m))  / (alpha * omega_m * beta)
    c_1 =  2/b
    c_2 =  2/(aux * b)
    return c_1, c_2



def params_fisicos_to_modelo_varios_n(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2'''
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = ((100/c_luz_norm)**2 * 6 * (1 - omega_m))  / (alpha * omega_m * beta)
    c_2 =  (2/(aux * b)) ** n
    c_1 =  aux * c_2
    return c_1, c_2

#%%
if __name__ == '__main__':
    omega_m_true = 0.24
    b_true = 2
    H_0=73.48

    c1,c2 = params_fisicos_to_modelo(omega_m_true, b_true)
    print(c1,c2)

    #c1_true = 1
    #c2_true = 1/19
    print(1/19)
