import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_LambdaCDM import H_LCDM
from funciones_int import Hubble_teorico
from funciones_int_sist_1 import Hubble_teorico_1
from funciones_int_sist_2 import Hubble_teorico_2

from funciones_BAO import r_drag, Ds_to_obs_final

#%%
def magn_aparente_teorica(zs, Hs, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(Hs**(-1), zs, initial=0)
    dc_int = interp1d(zs, d_c)(zcmb) #Interpolamos
    d_L = (1 + zhel) * dc_int #Obs, Caro multiplica por Zhel, con Zcmb da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi2_supernovas(muth, muobs, C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''

    deltamu = muth - muobs #vector fila
    transp = np.transpose(deltamu) #vector columna
    aux = np.dot(C_invertida,transp) #vector columna
    chi2 = np.dot(deltamu,aux) #escalar
    return chi2

def Hs_to_Ds(zs, Hs, z_data, index):
    if index == 4: #H
        aux = Hs
    elif index == 1: #DH
        DH = c_luz_km * (Hs**(-1))
        aux = DH
    else:
        INT = cumtrapz(Hs**(-1), zs, initial=0)
        DA = (c_luz_km/(1 + zs)) * INT
        if index == 0: #DA
            aux = DA
        elif index == 2: #DM
            #aux = (1+zs) * DA
            DM = c_luz_km * INT
            aux = DM
        elif index == 3: #DV
            #aux = (((1 +zs) * DA)**2 * c_luz_km * zs * (Hs**(-1))) ** (1/3)
            DV = c_luz_km * (INT**2 * zs * (Hs**(-1))) ** (1/3)
            aux = DV
    output = interp1d(zs,aux)
    return output(z_data)


def zs_2_logDlH0(zs,Es,z_data):
    INT = cumtrapz(Es**(-1), zs, initial=0)
    DlH0 = (c_luz_km * (1 + zs)) * INT #km/seg
    output = interp1d(zs,DlH0)
    return np.log10(output(z_data)) #log(km/seg)



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


def params_to_chi2(theta, params_fijos, index=0,
                    dataset_SN=None, dataset_CC=None,
                    dataset_BAO=None, dataset_AGN=None, H0_Riess=False,
                    cantidad_zs=int(10**5), model='HS',n=1,
                    nuisance_2 = False, errores_agrandados=False,
                    integrador=0, all_analytic=False):
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
    if integrador==0:
        zs_modelo, Hs_modelo = Hubble_teorico(params_fisicos, n=n, model=model,
                                    z_min=0, z_max=10, cantidad_zs=cantidad_zs,
                                    all_analytic=all_analytic)
                                    #Los datos de AGN van hasta z mas altos!
    elif integrador==1:
        zs_modelo, Hs_modelo = Hubble_teorico_1(params_fisicos, n=n, model=model,
                                    z_min=0, z_max=10, cantidad_zs=cantidad_zs,
                                    all_analytic=all_analytic)
                                    #Los datos de AGN van hasta z mas altos!
    elif integrador==2:
        zs_modelo, Hs_modelo = Hubble_teorico_2(params_fisicos, n=n, model=model,
                                    z_min=0, z_max=10, cantidad_zs=cantidad_zs,
                                    all_analytic=all_analytic)
                                    #Los datos de AGN van hasta z mas altos!

    #MAL!
    #Filtro para z=0 para que no diverja la integral de (1/H)
    #mask = zs_modelo_2 > 0.001
    #zs_modelo = zs_modelo_2[mask]
    #Hs_modelo = Hs_modelo_2[mask]


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

        if nuisance_2 == True:
            beta = 8.513
            ebeta = 0.437
            gamma = 0.622
            egamma = 0.014
        elif errores_agrandados == True:
            beta = 7.735
            ebeta = 2.44
            gamma = 0.648
            egamma = 0.07
        else: #Caso Estandar
            beta = 7.735
            ebeta = 0.244
            gamma = 0.648
            egamma = 0.007

        Es_modelo = Hs_modelo/H_0
        DlH0_teo = zs_2_logDlH0(zs_modelo,Es_modelo,z_data)
        DlH0_obs =  np.log10(3.24) - 25 + (logFx - gamma * logFuv - beta) / (2*gamma - 2)

        df_dgamma =  (-logFx+beta+logFuv) / (2*(gamma-1)**2)
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
    a = params_to_chi2([-19.37, 0.1, 80], 0.1, index=32,
                    #dataset_SN = ds_SN,
                    dataset_CC = ds_CC,
                    dataset_BAO = ds_BAO,
                    dataset_AGN = ds_AGN,
                    #H0_Riess = True,
                    model = 'HS'
                    )
    print(a)
