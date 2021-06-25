"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m, b, H_0, n


def params_fisicos_to_modelo_HS(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2. Notar que no depende de H0!'''
    aux = c_luz_km**2 * omega_m / (7800 * (8315)**2 * (1-omega_m)) #B en la tesis
    if n==1:
        c_1 =  2/b
        c_2 = 2*aux/b

    else:
        c_2 =  (2*aux/b) ** n
        c_1 =  c_2/aux
    return c_1, c_2

def params_fisicos_to_modelo_ST(omega_m, b, H0):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b
    y la constante de Hubble H0) y devuelve los parametros del modelo de
    Starobinsky lambda y Rs. Notar que no depende de n!'''
    Lambda = 3 * (H0/c_luz_km)**2 * (1-omega_m) #Constante cosmológica
    lamb = 2/b
    Rs = Lambda * b
    return lamb, Rs


#%%
if __name__ == '__main__':
#Testeamos Hu-Sawicki
    omega_m_true = 0.24
    b_true = 2
    H_0=73.48

    c1,c2 = params_fisicos_to_modelo_HS(omega_m_true, b_true,n=1)
    print(c1,c2)

    #c1_true = 1
    #c2_true = 1/19
    print(1/19)

#1.0 0.05262837317249587 #nuevo
#1.0 0.05262837317249588 #viejo
#0.05263157894736842 #1/19



#%%
    aux = c_luz_km**2 * omega_m_true / (7800 * (8315)**2 * (1-omega_m_true)) #B en la tesis
    aux

#%% Testeamos Starobinsky
    lamb,Rs = params_fisicos_to_modelo_ST(omega_m_true, b_true, H_0)
    print(lamb,Rs)
