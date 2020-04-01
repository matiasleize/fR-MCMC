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
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo


#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

def chi_2_cronometros(H_data, H_teo, dH):
    chi2 = np.sum(((H_data-H_teo)/dH)**2)
    return chi2

def params_to_chi2_H0_fijo(cond_iniciales, theta, params_fijos, z_data, H_data,
                            dH, cantidad_zs=3000, max_step=0.01, verbose=True):
    '''Dados los parámetros libres del modelo (omega,b) y los que quedan params_fijos (H_0,n),
    devuelve un chi2 para los datos de los cronómetros cósmicos'''

    [omega_m,b] = theta
    [H_0,n] = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b,H_0,n)
    params_modelo = [c1,r_hs/(H_0**2),c2,n] #de la cruz: [b,c,d,n]
    z,E = integrador(cond_iniciales, params_modelo, cantidad_zs=cantidad_zs,
                    max_step=max_step, verbose=verbose)
    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi

def params_to_chi2(cond_iniciales, theta, params_fijos, z_data, H_data, dH,
                    , cantidad_zs=3000, max_step=0.01, verbose=True):
    '''Dados los parámetros libres del modelo (omega, b y H0) y los que quedan params_fijos (n),
    devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [omega_m,b,H_0] = theta
    n = params_fijos

    ## Transformo los parametros fisicos en los del modelo:
    c1,c2,r_hs = params_fisicos_to_modelo(omega_m,b,H_0,n)
    params_modelo = [c1,r_hs/(H_0**2),c2,n] #de la cruz: [b,c,d,n]
    z,E = integrador(cond_iniciales, params_modelo, cantidad_zs=cantidad_zs,
                    max_step=max_step,verbose=verbose)#100,0.1

    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi


def params_to_chi2_viejos(cond_iniciales, theta, params_fijos, z_data, H_data,
                            dH, cantidad_zs=3000, max_step=0.01, verbose=True):
    '''Dados los parámetros libres del modelo (cq,c2,c0,H0) y los que quedan params_fijos (n),
    devuelve un chi2 para los datos de los
    cronómetros cósmicos'''

    [c1,c2,c0,H_0] = theta
    n = params_fijos
    ## Transformo los parametros fisicos en los del modelo:
    params_modelo = [c1,c0,c2,n] #de la cruz: [b,c,d,n]
    z,E = integrador(cond_iniciales, params_modelo, cantidad_zs=cantidad_zs,
                    max_step=max_step,verbose=verbose)#100,0.1

    H_int = interp1d(z,E)
    H_teo = H_0 * H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    return chi
