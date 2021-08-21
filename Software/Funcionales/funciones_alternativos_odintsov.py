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
        [Mabs, omega_m, H_0] = theta
        b = params_fijos

    elif index == 33:
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


def params_to_chi2_odintsov(theta, params_fijos, index=0,
                    dataset_SN=None, dataset_CC=None,
                    dataset_BAO=None, dataset_AGN=None, H0_Riess=False,
                    cantidad_zs=int(10**5), model='HS',n=1,
                    nuisance_2 = False, errores_agrandados=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas.'''

#    chi2_SN = chi2_CC = chi2_BAO = chi2_AGN = chi2_H0 =  0

    chi2_SN = 0
    chi2_CC = 0
    chi2_BAO = 0
    chi2_AGN = 0
    chi2_H0 =  0

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
        z_data_BAO, H_data_BAO, dH_BAO, rd_fid = dataset_BAO
        H_interp = interp1d(zs_modelo, Hs_modelo)
        H_teo = H_interp(z_data_BAO)

        H_data_BAO_norm = np.zeros(len(H_data_BAO))
        for i in range(len(H_data_BAO_norm)):
            if rd_fid[i]==1:
                factor = 1
            else:
                rd = r_drag(omega_m,H_0,wb=0.0225) #Calculo del rd, fijo wb!! CHequear que es correcto
                factor = rd_fid[i]/rd
            H_data_BAO_norm[i] = H_data_BAO[i] * factor
        chi2_BAO = chi2_sin_cov(H_teo,H_data_BAO_norm,dH_BAO**2)

    return chi2_SN + chi2_CC + chi2_BAO

#%%
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon, leer_data_cronometros, leer_data_BAO_odintsov

    # Supernovas
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    ds_SN = leer_data_pantheon('lcparam_full_long_zhel.txt')

    # Cronómetros
    os.chdir(path_git+'/Software/Estadística/Datos/')
    ds_CC = leer_data_cronometros('datos_cronometros.txt')

    # BAO de odintsov
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_odintsov')
    ds_BAO = leer_data_BAO_odintsov('datos_bao_odintsov.txt')
#%%
    a = params_to_chi2_odintsov([-19.351100617405038, 0.30819459447582237, 69.2229987565787], _, index=32,
                    dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    #dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'LCDM'
                    )
    print(a)
