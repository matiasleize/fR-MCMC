"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_int import integrador
from funciones_taylor import Taylor_HS, Taylor_ST

from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

#Funciones de croómetros
def chi_2_cronometros(H_teo,H_data, dH):
    chi2 = np.sum(((H_data-H_teo)/dH)**2)
    return chi2

#Funciones de supernovas
def magn_aparente_teorica(zs, Hs, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(Hs**(-1), zs, initial=0)
    dc_int = interp1d(zs, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi_2_supernovas(muth,magn_aparente_obs,M_abs,C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''
    sn = len(muth)
    muobs =  magn_aparente_obs - M_abs
    deltamu = muobs - muth
    transp = np.transpose(deltamu)
    aux = np.dot(C_invertida,deltamu)
    chi2 = np.dot(transp,aux)
    return chi2

#Funciones BAO
def Hs_to_Ds(zs, Hs, z_data, index):
    INT = cumtrapz(Hs**(-1), zs, initial=0)
    DA = (c_luz_km/(1 +zs)) * INT

    if index == 0: #DA
        aux = DA

    if index == 1: #DH
        aux = c_luz_km/Hs

    if index == 2: #DM
        aux = (1+zs) * DA

    if index == 3: #DV
        aux = ((1 +zs)**2 * DA**2 * (c_luz_km * zs/Hs)) ** (1/3)

    if index == 4: #H
        aux = Hs

    output = interp1d(zs,aux)
    return output(z_data)

def chi_2_BAO(teo, data, errores):
    chi2 = np.sum(((data-teo)/errores)**2)
    return chi2

#Juntamos todo
def params_to_chi2_taylor(theta, params_fijos, zcmb, zhel, Cinv,
                    mb,z_data_cron, H_data_cron, dH_cron, dataset_BAO,
                    cantidad_zs=int(10**5), verbose=True, num_datasets=5,
                    M_fijo=False, model='HS',chi_riess=True):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if len(theta)==4:
        [Mabs,omega_m,b,H_0] = theta
        n = params_fijos
        chi2_H0 = ((H_0-73.48)/1.66)**2
    elif len(theta)==3:
        if M_fijo==True:
            [omega_m,b,H_0] = theta
            [Mabs,n] = params_fijos
            chi2_H0 = ((H_0-73.48)/1.66)**2
        else:
            [Mabs,omega_m,b] = theta
            [H_0,n] = params_fijos
            chi2_H0 = 0

    #Cronometros:
    if model=='HS':
        H_modelo_cron = Taylor_HS(z_data_cron, omega_m, b, H_0)
    else:
        H_modelo_cron = Taylor_ST(z_data_cron, omega_m, b, H_0)



    if chi_riess==True:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)+chi2_H0
    else:
        chi_cron = chi_2_cronometros(H_modelo_cron,H_data_cron,dH_cron)


    #Supernovas:
    zs_SN = np.linspace(0,3,cantidad_zs)

    if model=='HS':
        H_modelo_SN = Taylor_HS(zs_SN, omega_m, b, H_0)
    else:
        H_modelo_SN = Taylor_ST(zs_SN, omega_m, b, H_0)

    muth = magn_aparente_teorica(zs_SN,H_modelo_SN,zcmb,zhel)
    chi_sn = chi_2_supernovas(muth,mb,Mabs,Cinv)

    #BAO:
    zs_BAO = np.linspace(0.1, 3, cantidad_zs)
    if model=='HS':
        H_modelo_BAO = Taylor_HS(zs_BAO, omega_m, b, H_0)
    else:
        H_modelo_BAO = Taylor_ST(zs_BAO, omega_m, b, H_0)
    h = H_0/100
    #omega_b = 0.05 #(o algo asi)
    cte = 0.02 #(omega_B * h**2)

    #Calculo el zd
    b1 = 0.313*(omega_m*h**2)**(-0.419)*(1+0.607*(omega_m*h**2)**(0.6748))
    b2 = 0.238*(omega_m*h**2)**0.223
    zd = ((1291*(omega_m*h**2)**0.251)/(1+0.659*(omega_m*h**2)**0.828)) * (1+b1*cte**b2)

    #Calculo del rd:
    zs_int = np.logspace(np.log10(zd),13,int(10**5))
    if model=='HS':
        H_int = Taylor_HS(zs_int, omega_m,b,H_0)
    else:
        H_int = Taylor_ST(zs_int, omega_m, b, H_0)
    R_bar = 31500 * cte * (2.726/2.7)**(-4)
    #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    #integrando = cs/H_int
    integrando = c_luz_km / (H_int * np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    rd = simps(integrando,zs_int)

    chies_BAO = np.zeros(num_datasets)

    for i in range(num_datasets):
        (z_data, valores_data, errores_data, rd_fid) = dataset_BAO[i]
        #print(type(valores_data))
        if isinstance(z_data,np.float64):
            if (rd_fid == 1):
                pass
            else:
                valores_data = valores_data #* (rd/rd_fid)
                errores_data = errores_data #* (rd/rd_fid)
        elif isinstance(z_data,np.ndarray):
            for j in range(len(z_data)):
                if (rd_fid[j] == 1):
                        pass
                else:
                    valores_data[j] = valores_data[j] #*(rd/rd_fid[j])
                    errores_data[j] = errores_data[j]# *(rd/rd_fid[j])
        outs = Hs_to_Ds(zs_BAO,H_modelo_BAO,z_data,i)
        chies_BAO[i] = chi_2_BAO(outs,valores_data,errores_data)
    if np.isnan(sum(chies_BAO))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)
    chi_BAO = np.sum(chies_BAO)

    return chi_cron+chi_sn+chi_BAO
