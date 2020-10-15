"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d


import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_taylor import Taylor_HS, Taylor_ST
from funciones_condiciones_iniciales import condiciones_iniciales


#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

def chi_2_cronometros(H_teo,H_data, dH):
    chi2 = np.sum(((H_data-H_teo)/dH)**2)
    return chi2


def params_to_chi2(theta, params_fijos, z_data, H_data, dH,
                    cantidad_zs=10000, max_step=0.005,
                    verbose=True, model='HS',
                    chi_riess=True, taylor=False, H0_nunes=False):
    '''Dados los parámetros libres del modelo (omega, b y H0) y los que quedan params_fijos (n),
    devuelve un chi2 para los datos de los
    cronómetros cósmicos'''
    if len(theta)==3:
        [omega_m,b,H_0] = theta
        n = params_fijos
        if H0_nunes == True:
            chi_H0 = ((H_0-73.02)/1.79)**2
        else:
            chi_H0 = ((H_0-73.48)/1.66)**2
    elif len(theta)==2:
        [omega_m,b] = theta
        [H_0,n] = params_fijos
        chi_H0 = 0

    if taylor == True:
        if model=='HS':
            H_teo = Taylor_HS(z_data, omega_m, b, H_0)
        else:
            H_teo = Taylor_ST(z_data, omega_m, b, H_0)

    else:
        if (0 <= b < 0.2):
            if model=='HS':
                H_teo = Taylor_HS(z_data, omega_m, b, H_0)
            else:
                H_teo = Taylor_ST(z_data, omega_m, b, H_0)
        else:
            params_fisicos = [omega_m,b,H_0]
            z, H = integrador(params_fisicos, n)
            H_int = interp1d(z,H)
            H_teo = H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)


    if chi_riess==True:
        return chi+chi_H0
    else:
        return chi
