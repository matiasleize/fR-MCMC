"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time

from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
#%%

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

#Supernovas

def magn_aparente_teorica(z, H, zcmb, zhel):
    '''A partir de un array de redshift y un array de la magnitud E = H_0/H
    que salen de la integración numérica, se calcula el mu teórico que deviene
    del modelo. muth = 25 + 5 * log_{10}(d_L),
    donde d_L = (c/H_0) (1+z) int(dz'/E(z'))'''

    d_c =  c_luz_km * cumtrapz(H**(-1), z, initial=0)
    dc_int = interp1d(z, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
    #Magnitud aparente teorica
    muth = 25.0 + 5.0 * np.log10(d_L)
    return muth

def chi_2_supernovas(muth, muobs, C_invertida):
    '''Dado el resultado teórico muth y los datos de la
    magnitud aparente y absoluta observada, con su matriz de correlación
    invertida asociada, se realiza el cálculo del estadítico chi cuadrado.'''

    deltamu = muth - muobs #vector fila
    transp = np.transpose(deltamu) #vector columna
    aux = np.dot(C_invertida,transp) #vector columna
    chi2 = np.dot(deltamu,aux) #escalar
    return chi2


def params_to_chi2(theta, params_fijos, zcmb, zhel, Cinv, mb,
                    cantidad_zs=int(10**5), fix_M=False,
                     fix_H0=False, fix_M_H0=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas.'''

    if fix_M == True:
        [omega_m, H_0] = theta
        Mabs = params_fijos
    elif fix_H0 == True:
        [Mabs, omega_m] = theta
        H_0 = params_fijos
    elif fix_M_H0 == True:
        omega_m = theta
        [Mabs, H_0] = params_fijos
    else:
        [Mabs, omega_m, H_0] = theta

    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)

    muth = magn_aparente_teorica(z, H, zcmb, zhel)
    muobs =  mb - Mabs

    chi = chi_2_supernovas(muth, muobs, Cinv)
    return chi


# Cronómetros
def chi_2_cronometros(H_teo, H_data, dH):
    chi2 = np.sum(((H_data - H_teo) / dH)**2)
    return chi2

def params_to_chi2_cronometros(theta, z_data, H_data,
                                dH, cantidad_zs=int(10**6)):
    '''Dados los parámetros libres del modelo (omega y H0), devuelve un chi2
     para los datos de los cronómetros cósmicos'''

    [omega_m, H_0] = theta
    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)
    H_int = interp1d(z, H)
    H_teo = H_int(z_data)
    chi = chi_2_cronometros(H_teo, H_data, dH)
    return chi

# BAO
def Hs_to_Ds(zs, Hs, z_data, index):
    INT = cumtrapz(Hs**(-1), zs, initial=0)
    DA = (c_luz_km/(1 + zs)) * INT
    if index == 0: #DA
        aux = DA
    elif index == 1: #DH
        aux = c_luz_km * (Hs**(-1))
    elif index == 2: #DM
        #aux = (1+zs) * DA
        aux = c_luz_km * INT
    elif index == 3: #DV
        #aux = (((1 +zs) * DA)**2 * c_luz_km * zs * (Hs**(-1))) ** (1/3)
        aux = c_luz_km * (INT**2 * zs * (Hs**(-1))) ** (1/3)
    elif index == 4: #H
        aux = Hs
    output = interp1d(zs,aux)
    return output(z_data)


def chi_2_BAO(teo, data, errores):
    chi2 = np.sum(((data-teo)/errores)**2)
    return chi2

def params_to_chi2_BAO(theta, params_fijos, dataset,
                        cantidad_zs=int(10**6),num_datasets=5):
    '''Dados los parámetros libres del modelo (omega, b y H0) y
    los que quedan params_fijos (n), devuelve un chi2 para los datos
    de BAO'''

    [omega_m, H_0] = theta

    zs = np.linspace(0.01, 3, cantidad_zs)
    H_teo = H_LCDM(zs, omega_m, H_0)

    h = H_0/100
    omega_b = 0.05 #(o algo asi)
    cte = 0.02 #(omega_B * h**2)

    #Calculo del rd:
    zs_int = np.linspace(1100,3600,100000)
    H_int = H_LCDM(zs_int, omega_m, H_0)
    R_bar = 31500 * cte * (2.726/2.7)**(-4)
    #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    #integrando = cs/H_int
    integrando = c_luz_km / (H_int * np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    rd = simps(integrando,zs_int)

    chies = np.zeros(num_datasets)

    for i in range(num_datasets):
        (z_data, valores_data, errores_data, rd_fid) = dataset[i]
        #print(type(valores_data))
        if isinstance(z_data,np.float64):
            if (rd_fid == 1):
                pass
            else:
                valores_data = valores_data * (rd/rd_fid)
                errores_data = errores_data * (rd/rd_fid)
        elif isinstance(z_data,np.ndarray):
            for j in range(len(z_data)):
                if (rd_fid[j] == 1):
                        pass
                else:
                    valores_data[j] = valores_data[j] * (rd/rd_fid[j])
                    errores_data[j] = errores_data[j] * (rd/rd_fid[j])
        outs = Hs_to_Ds(zs,H_teo,z_data,i)
        chies[i] = chi_2_BAO(outs,valores_data,errores_data)
    if np.isnan(sum(chies))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)

    return np.sum(chies)
