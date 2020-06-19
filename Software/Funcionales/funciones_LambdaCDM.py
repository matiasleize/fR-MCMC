"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time

from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg
#%%

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

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
                    cantidad_zs=int(10**5), Mfijo=False, omega_solo=False):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if Mfijo == True:
        [omega_m, H_0] = theta
        Mabs = params_fijos
    elif omega_solo == True:
        omega_m = theta
        [Mabs, H_0] = params_fijos
    else:
        [Mabs, omega_m] = theta
        H_0 = params_fijos

    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)

    muth = magn_aparente_teorica(z, H, zcmb, zhel)
    muobs =  mb - Mabs

    chi = chi_2_supernovas(muth, muobs, Cinv)
    return chi

def params_to_chi2_3params(theta, zcmb, zhel, Cinv, mb, cantidad_zs=10000):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

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
                                dH, cantidad_zs=3000):
    '''Dados los parámetros libres del modelo (omega y H0), devuelve un chi2
     para los datos de los cronómetros cósmicos'''

    [omega_m, H_0] = theta
    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)
    H_int = interp1d(z, H)
    H_teo = H_int(z_data)
    chi = chi_2_cronometros(H_teo, H_data, dH)
    return chi

#%%
if __name__ == '__main__':
    ##%matplotlib qt5
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()

    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon

    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    zcmb, zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')

    z = np.linspace(0,3,1000)
    H_0 = 73.48
    #omega_m = 0.3

    M_abs = -18
    muobs =  mb - M_abs
    plt.figure()
    plt.errorbar(zcmb,muobs, fmt='.k',label='observado')

    for omega_m in (np.linspace(0,1,10)):
        H = H_LCDM(z,omega_m,H_0)
        #plt.plot(z,H_0 * E)
        #aux = np.zeros(len(E))
        d_c = cumtrapz(1/H,z,initial=0) #Paso c_luz a km/seg
        dc_int = interp1d(z,d_c) #Interpolamos
        d_L = (1+zhel) * dc_int(zcmb) #Obs, Caro multiplica por Zhel, con Zobs da un poquin mejor
        ##Magnitud aparente teorica
        muth = 25.0 + 5.0 * np.log10(d_L)
        sn = len(muth)
        plt.plot(zcmb,muth,'.',label='omega={}'.format(omega_m))
        deltamu = muobs - muth
        plt.legend(loc='best')

        print(muth)
        #%%
        plt.figure()
        z = np.linspace(0,0.5,1000)
        for omega_m in (np.linspace(0,1,10)):
            H = H_LCDM(z,omega_m,H_0)
            plt.plot(z, H)
            plt.plot(z,H,label='omega={}'.format(omega_m))
            plt.legend(loc='best')
#%%
        plt.close()
        plt.figure()
        z = np.linspace(0,3,int(10**6))
        for omega_m in (np.linspace(0.2,0.4,3)):
            H = H_LCDM(z,omega_m,H_0=1)
            aux = cumtrapz(H**(-1),z,initial=0)
            plt.plot(z, aux,label='omega={}'.format(omega_m))
            #d_L = (1+z) * 0.001 * c_luz * cumtrapz(1/H,z,initial=0)
            #plt.plot(z, d_L)
            #plt.plot(z,H,label='omega={}'.format(omega_m))
            plt.legend(loc='best')
#%%
    def integral(omega_m,z):
        zs = np.linspace(0,z,100000)
        H = H_LCDM(zs,omega_m,H_0=1)
        aux = cumtrapz(H**(-1),zs,initial=0)
        return aux[-1]

    Z = 0.01
    omega_m=0.2
    H = H_LCDM(z,omega_m,H_0=1)
    aux = cumtrapz(H**(-1),z,initial=0)
    dc_int = interp1d(z,aux) #Interpolamos
    #%%
    print(dc_int(Z))
    #%%
    dc = np.zeros(len(zcmb))
    for i,z in enumerate (zcmb):
        dc[i]=integral(0.2,z)
    plt.plot(zcmb,dc_int(zcmb),'.')
    plt.plot(zcmb,dc+0.001,'.')

    #plt.figure()
    #plt.plot(zcmb,deltamu,'.')
    #plt.plot(zcmb,(muth-muobs)/muobs,'r.')
#%%

    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_pantheon
    from funciones_LambdaCDM import params_to_chi2
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
    zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
    #caro
    theta = [-19.38,0.299]
    params_fijos = 69.01
    #minimizacion
    theta = [-19.380657833554064, 0.2981498895755162]
    params_fijos = 69.01
    #0.9798300382238752

    params_to_chi2(theta,params_fijos,zcmb, zhel, Cinv, mb,
                        cantidad_zs=int(10**6))
