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
#from funciones_int import Hubble_teorico
from funciones_int_sist_1 import Hubble_teorico_1
from funciones_LambdaCDM import H_LCDM

#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

def zs_2_logDlH0(INT,z_data):
    DlH0 = (c_luz_km * (1 + z_data)) * INT #km/seg
    return np.log10(DlH0) #log(km/seg)

### Nuisance AGN
def Hs_2_logDl(zs,Hs,z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)

    Dl = (c_luz_km * (1 + zs)) * INT #Mpc
    output = interp1d(zs,Dl)
    return np.log10(output(z_data)) #log(Mpc)

def Hs_2_logDlH0(zs,Hs,z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)

    Dl = (c_luz_km * (1 + zs)) * H0 * INT #Mpc
    output = interp1d(zs,Dl)
    return np.log10(output(z_data)) #log(Mpc)



def chi2_AGN_nuisance(teo, data, errores_cuad):
    chi2 = np.sum( ((data-teo)**2/errores_cuad) + np.log(2*np.pi*errores_cuad)) #o menos en el paper
    return chi2

def params_to_chi2_AGN_nuisance(theta, params_fijos, dataset_AGN, n=1,
                                cantidad_zs=int(10**6), model='HS'
                                ,less_z=False,all_analytic=False):
    '''
    Dados los parámetros del modelo devuelvo el estadítico chi2 para
    los datos de AGN.
    '''
    #Defino los parámetros que voy a utilizar
    if model == 'LCDM':
        if isinstance(theta,float):
            #print(theta)
            omega_m = theta
            [beta, gamma, delta, H_0] = params_fijos
            zs_modelo = np.linspace(0,10,10**5)
            Hs_modelo = H_LCDM(zs_modelo,omega_m,H_0)


        else:
            if len(theta) == 4:
                [omega_m, beta, gamma, delta] = theta #Este beta es distinto al otro!
                H_0 = params_fijos
                zs_modelo = np.linspace(0,10,10**5)
                Hs_modelo = H_LCDM(zs_modelo,omega_m,H_0)


    else:
        if len(theta) == 5:
            [omega_m, b, beta, gamma, delta] = theta #Este beta es distinto al otro!
            H_0 = params_fijos
        elif len(theta) == 4:
            [omega_m, b] = theta #Este beta es distinto al otro!
            [beta, gamma, delta, H_0] = params_fijos

        params_fisicos = [omega_m,b,H_0]
        zs_modelo, Hs_modelo = Hubble_teorico_1(params_fisicos, n=n, model=model,
                                    z_min=0, z_max=10, cantidad_zs=cantidad_zs,
                                    all_analytic=all_analytic)


    #Importo los datos
    z_data_unmasked, logFuv_unmasked, eFuv_unmasked, logFx_unmasked, eFx_unmasked  = dataset_AGN

    if less_z == True:
        mask = z_data_unmasked<1.5
        z_data = z_data_unmasked[mask]
        logFuv = logFuv_unmasked[mask]
        eFuv = eFuv_unmasked[mask]
        logFx = logFx_unmasked[mask]
        eFx = eFx_unmasked[mask]
    else:
        z_data = z_data_unmasked
        logFuv = logFuv_unmasked
        eFuv = eFuv_unmasked
        logFx = logFx_unmasked
        eFx = eFx_unmasked


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

    #beta_true =  6.8#7.735
    gamma_true = 0.648
    delta_true = 0.235
    H0_true =  70


    for j,beta_true in enumerate(np.linspace(7.9,8.5,20)):
        params_fijos = [beta_true, gamma_true, delta_true, H0_true]
        omegas = np.linspace(0,1,50)
        chi_2 = np.zeros(50)
        for i,omega_m in enumerate(omegas):
            chi_2[i] = params_to_chi2_AGN_nuisance(omega_m, params_fijos,
                                data_agn, model='LCDM')
        plt.figure()
        plt.plot(omegas, chi_2)
        plt.grid()
        plt.title(r'$\beta$ = {}'.format(beta_true))
        plt.ylabel('$\chi^{2}$')
        plt.xlabel('$\Omega_{m}$')
        plt.savefig('/home/matias/Desktop/chi_2__beta={}.png'.format(beta_true))
        plt.close()
