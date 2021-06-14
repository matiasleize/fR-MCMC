"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps

from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_data import leer_data_BAO

#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

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


def params_to_chi2_taylor(theta, params_fijos, dataset, cantidad_zs=int(10**5),
                        num_datasets=5,model='HS'):
    '''Dados los parámetros libres del modelo (omega, b y H0) y
    los que quedan params_fijos (n), devuelve un chi2 para los datos
    de BAO'''

    if len(theta)==3:
        [omega_m,b,H_0] = theta
        n = params_fijos
    elif len(theta)==2:
        [omega_m,b] = theta
        [H_0,n] = params_fijos

    zs_modelo = np.linspace(0, 3, cantidad_zs) #Mas chico diverge, esta biene esto xq el
    #primer z esta en 0.106 y el paso es del orden de 10**(-5)

    if model=='HS':
        H_modelo = Taylor_HS(zs, omega_m, b,H_0)
    else:
        H_modelo = Taylor_ST(zs, omega_m, b,H_0)

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
        H_int = Taylor_ST(zs_int, omega_m,b,H_0)
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
                valores_data = valores_data #* (rd/rd_fid)
                errores_data = errores_data #* (rd/rd_fid)
        elif isinstance(z_data,np.ndarray):
            for j in range(len(z_data)):
                if (rd_fid[j] == 1):
                        pass
                else:
                    valores_data[j] = valores_data[j] #*(rd/rd_fid[j])
                    errores_data[j] = errores_data[j]# *(rd/rd_fid[j])
        outs = Hs_to_Ds(zs_modelo,H_modelo,z_data,i)
        chies[i] = chi_2_BAO(outs,valores_data,errores_data)
    if np.isnan(sum(chies))==True:
        print('Hay errores!')
        print(omega_m,b,H_0,rd)

    return np.sum(chies)

#%%
if __name__ == '__main__':
    from scipy.constants import c as c_luz #metros/segundos
    c_luz_km = c_luz/1000
    def H_LCDM(z, omega_m, H_0):
        omega_lambda = 1 - omega_m
        H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
        return H

    os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
    dataset = []
    archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivo_BAO[i])
        dataset.append(aux)

    dataset[1][0][1] #num. dataset / tipo de dato / lugar del dato
    [omega_m,b,H_0,rd,n]=[0.3,0.1,73.48,1,1]
    cantidad_zs = 10000
    #lala = params_to_chi2_taylor([omega_m,b,H_0,rd], n, dataset)
    #print(lala)

    zs = np.linspace(0,3,cantidad_zs)
    #H_teo = Taylor_HS(zs, omega_m, b, H_0)
    H_teo = H_LCDM(zs, omega_m, H_0)

    #Calculo del rd:
    zs_int = np.logspace(np.log10(1110),13,1000) #1110
    H_int = H_LCDM(zs_int, omega_m, H_0)
    cte = 0.02 #h**2 * omega_b
    R_bar = 31500 * cte * (2.716/2.7)**(-4)
    cs = c_luz_km/np.sqrt(3*(1+R_bar*(1+zs_int)**(-1)))
    integrando = cs/H_int
    rd = simps(integrando,zs_int)
    print(rd)

#%%
    num_datasets = 5
    chies = np.zeros(num_datasets)
    for i in range(num_datasets):
        (z_data, valores_data, errores_data, rd_bool) = dataset[i]
        print(type(valores_data))
        if isinstance(z_data,np.float64):
            if rd_bool == 1:
                valores_data = valores_data * rd
                errores_data = errores_data * rd
            else:
                pass
        else:
            for j in range(len(z_data)):
                if rd_bool[j] == 1:
                    valores_data[j] = valores_data[j] * rd
                    errores_data[j] = errores_data[j] * rd
                else:
                    pass
        #print(valores_data)
        outs = Hs_to_Ds(zs,H_teo,z_data,i)
        chies[i] = chi_2_BAO(outs,valores_data,errores_data)
    print(chies)
    print(outs)
    len(dataset[4][0])
    dataset

    print(np.sum(chies))
