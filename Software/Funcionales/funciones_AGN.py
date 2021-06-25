"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz

from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import Hubble_teorico
from funciones_LambdaCDM import E_LCDM, H_LCDM

#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

def zs_2_logDlH0(zs,omega_m,z_data):
    Es = E_LCDM(zs, omega_m)
    INT = cumtrapz(Es**(-1), zs, initial=0)
    DlH0 = (c_luz_km * (1 + zs)) * INT #km/seg
    output = interp1d(zs,DlH0)
    return np.log10(output(z_data)) #log(km/seg)

### Nuisance AGN

def Hs_2_logDl(zs,Hs,z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)

    Dl = (c_luz_km * (1 + zs)) * INT #Mpc
    output = interp1d(zs,Dl)
    return np.log10(output(z_data)) #log(Mpc)

def chi2_AGN_nuisance(teo, data, errores_cuad):
    chi2 = np.sum( ((data-teo)**2/errores_cuad) + np.log(2*np.pi*errores_cuad)) #o menos en el paper
    return chi2

def params_to_chi2_AGN_nuisance(theta, params_fijos, dataset_AGN, n=1,
                                cantidad_zs=int(10**6), model='HS'):
    '''
    Dados los parámetros del modelo devuelvo el estadítico chi2 para
    los datos de AGN.
    '''
    #Defino los parámetros que voy a utilizar
    if model == 'LCDM':
        if len(theta) == 4:
            [omega_m, beta, gamma, delta] = theta #Este beta es distinto al otro!
            [b,H_0] = params_fijos

    else:
        if len(theta) == 5:
            [omega_m, b, beta, gamma, delta] = theta #Este beta es distinto al otro!
            H_0 = params_fijos
        elif len(theta) == 4:
            [omega_m, b, gamma, delta] = theta #Este beta es distinto al otro!
            [H_0, beta] = params_fijos

    params_fisicos = [omega_m,b,H_0]
    zs_modelo_2, Hs_modelo_2 = Hubble_teorico(params_fisicos, n=n, model=model,
                            z_min=0,z_max=10)

    #Filtro para z=0 para que no diverja la integral de (1/H)
    mask = zs_modelo_2 > 0.001
    zs_modelo = zs_modelo_2[mask]
    Hs_modelo = Hs_modelo_2[mask]

    #Importo los datos
    z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN

    Dl_teo = Hs_2_logDl(zs_modelo,Hs_modelo,z_data) #Mpc
    Dl_teo_cm = Dl_teo - np.log10(3.24) + 25
    psi = beta + gamma * logFuv + 2 * (gamma-1) * (Dl_teo_cm + 0.5 * np.log10(4*np.pi))

    si_2 = eFx**2 + gamma**2 * eFuv**2 + np.exp(2*np.log(delta)) #El cuadrado de los errores

    chi2_AGN = chi2_AGN_nuisance(psi, logFx, si_2)

    return chi2_AGN

#%%
if __name__ == '__main__':
    #Data AGN
    from funciones_data import leer_data_AGN
    from matplotlib import pyplot as plt
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
    data_agn = leer_data_AGN('table3.dat')

    theta = [0.99,0.1,8.3,0.4,0.2]
    params_fijos = [70,1]
    chi_2_AGN_nuisance = params_to_chi2_AGN_nuisance(theta, params_fijos, data_agn, model='EXP')
    print(chi_2_AGN_nuisance)