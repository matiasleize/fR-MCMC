"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import Hubble_teorico
from funciones_supernovas import magn_aparente_teorica, chi2_supernovas
from funciones_BAO import r_drag, Hs_to_Ds, Ds_to_obs_final
from funciones_AGN import zs_2_logDlH0

### Generales
def chi2_sin_cov(teo, data, errores_cuad):
    chi2 = np.sum((data-teo)**2/errores_cuad)
    return chi2

def all_parameters(theta, params_fijos, index):
    '''Esta función junta los valores de los parámetros
    variables y los parámetros fijos en una sola lista con un criterio
    dado por el valor de index.'''

    if index == 4:
        [Mabs, omega_m, b, H_0] = theta
        _ = params_fijos

    elif index == 31:
        [omega_m, b, H_0] = theta
        Mabs = params_fijos

    elif index == 32:
        [Mabs, omega_m, b] = theta
        H_0 = params_fijos

    elif index == 21:
        [omega_m, b] = theta
        [Mabs, H_0] = params_fijos

    elif index == 22:
        [omega_m, H_0] = theta
        [Mabs, b] = params_fijos

    elif index == 1:
        omega_m = theta
        [Mabs, b, H_0] = params_fijos


    return [Mabs, omega_m, b, H_0]


def params_to_chi2(theta, params_fijos, index=0,
                    dataset_SN=None, dataset_CC=None,
                    dataset_BAO=None, dataset_AGN=None, H0_Riess=False,
                    cantidad_zs=int(10**5), model='HS',n=1):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas.'''

    chi2_SN = chi2_CC = chi2_BAO = chi2_AGN = chi2_H0 =  0

    [Mabs, omega_m, b, H_0] = all_parameters(theta, params_fijos, index)

    params_fisicos = [omega_m,b,H_0]
    zs_modelo_2, Hs_modelo_2 = Hubble_teorico(params_fisicos, n=n, model=model,
                                z_min=0, z_max=10, cantidad_zs=cantidad_zs)
                                #Los datos de AGN van hasta z mas altos!

    #Filtro para z=0 para que no diverja la integral de (1/H)
    mask = zs_modelo_2 > 0.001
    zs_modelo = zs_modelo_2[mask]
    Hs_modelo = Hs_modelo_2[mask]


    if dataset_SN != None:
        #Importo los datos
        zcmb, zhel, Cinv, mb = dataset_SN
        muth = magn_aparente_teorica(zs_modelo, Hs_modelo, zcmb, zhel)
        muobs =  mb - Mabs
        chi2_SN = chi2_supernovas(muth, muobs, Cinv)

    if dataset_CC != None:
        #Importo los datos
        z_data, H_data, dH = dataset_CC
        H_interp = interp1d(zs_modelo, Hs_modelo)
        H_teo = H_interp(z_data)
        chi2_CC = chi2_sin_cov(H_teo, H_data, dH**2)

    if dataset_BAO != None:
        num_datasets=5
        chies_BAO = np.zeros(num_datasets)
        for i in range(num_datasets): #Para cada tipo de dato
            (z_data_BAO, valores_data, errores_data_cuad,wb_fid) = dataset_BAO[i]
            if i==0: #Dato de Da
                rd = r_drag(omega_m,H_0,wb_fid) #Calculo del rd
                distancias_teoricas = Hs_to_Ds(zs_modelo,Hs_modelo,z_data_BAO,i)
                output_teorico = Ds_to_obs_final(zs_modelo, distancias_teoricas, rd, i)
            else: #De lo contrario..
                distancias_teoricas = Hs_to_Ds(zs_modelo,Hs_modelo,z_data_BAO,i)
                output_teorico = np.zeros(len(z_data_BAO))
                for j in range(len(z_data_BAO)): #Para cada dato de una especie
                     rd = r_drag(omega_m,H_0,wb_fid[j]) #Calculo del rd
                     output_teorico[j] = Ds_to_obs_final(zs_modelo,distancias_teoricas[j],rd,i)
            #Calculo el chi2 para cada tipo de dato (i)
            chies_BAO[i] = chi2_sin_cov(output_teorico,valores_data,errores_data_cuad)

        if np.isnan(sum(chies_BAO))==True:
            print('Hay errores!')
            print(omega_m,H_0,rd)

        chi2_BAO = np.sum(chies_BAO)


    if dataset_AGN != None:
        #Importo los datos
        z_data, logFuv, eFuv, logFx, eFx  = dataset_AGN

        beta = 7.735
        ebeta = 0.244
        gamma = 0.648
        egamma = 0.007

        DlH0_teo = zs_2_logDlH0(zs_modelo,omega_m,z_data)
        DlH0_obs =  np.log10(3.24) - 25 + (logFx - gamma * logFuv - beta) / (2*gamma - 2)

        df_dgamma = (1/(2*(gamma-1))**2) * (-logFx+beta+logFuv)
        eDlH0_cuad = (eFx**2 + gamma**2 * eFuv**2 + ebeta**2)/ (2*gamma - 2)**2 + (df_dgamma)**2 * egamma**2 #El cuadrado de los errores

        chi2_AGN = chi2_sin_cov(DlH0_teo, DlH0_obs, eDlH0_cuad)

    if H0_Riess == True:
        chi2_H0 = ((H_0-73.48)/1.66)**2

    return chi2_SN + chi2_CC + chi2_AGN + chi2_BAO + chi2_H0

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO, leer_data_AGN

    # Supernovas
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

    # Cronómetros
    os.chdir(path_git+'/Software/Estadística/Datos/')
    ds_CC = leer_data_cronometros('datos_cronometros.txt')

    # BAO
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
    ds_BAO = []
    archivos_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivos_BAO[i])
        ds_BAO.append(aux)

    # AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
    ds_AGN = leer_data_AGN('table3.dat')

#%%
    params_to_chi2([0.3,0.1,72], [-19,1], index=0,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    dataset_AGN = ds_AGN,
                    H0_Riess = True,
                    model = 'EXP'
                    )
