"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import camb
from camb import model, initialpower
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
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
from funciones_BAO import r_drag, r_drag_camb
from funciones_LambdaCDM import H_LCDM
#%%

def chi_2_BAO(teo, data, errores):
    chi2 = np.sum(((data-teo)/errores)**2)
    return chi2


def Hs_to_Ds_old_data(zs, Hs, z_data, index):
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

def params_to_chi2_BAO_old_data(theta, params_fijos, dataset,
                        cantidad_zs=int(10**6),num_datasets=5):
    '''Dados los parámetros libres del modelo (omega, b y H0) y
    los que quedan params_fijos (n), devuelve un chi2 para los datos
    de BAO'''

    [omega_m, H_0] = theta
    zs_modelo = np.linspace(0.01, 3, cantidad_zs)
    H_modelo = H_LCDM(zs_modelo, omega_m, H_0)
    rd = r_drag_camb(omega_m,H_0) #Calculo del rd

    chies = np.zeros(num_datasets)

    for i in range(num_datasets):
        (z_data, valores_data, errores_data, rd_fid) = dataset[i]

        if i==0: #Dato de Da
            valores_data_2 = valores_data * (rd/rd_fid)
            errores_data_2 = errores_data * (rd/rd_fid)
            pass
        elif i==4: #Datos de H
            valores_data_2 = np.zeros(len(valores_data))
            errores_data_2 = np.zeros(len(errores_data))
            for j in range(len(z_data)):
                valores_data_2[j] = valores_data[j] * (rd/rd_fid[j])
                errores_data_2[j] = errores_data[j] * (rd/rd_fid[j])
        else:
            valores_data_2 = np.zeros(len(valores_data))
            errores_data_2 = np.zeros(len(errores_data))
            for j in range(len(z_data)):
                if rd_fid[j] != 1:
                    valores_data_2[j] = valores_data[j] * (rd_fid[j]/rd)
                    errores_data_2[j] = errores_data[j] * (rd_fid[j]/rd)
                else: #No hay que multiplicar x ningun factor
                    valores_data_2[j] = valores_data[j]
                    errores_data_2[j] = errores_data[j]

        outs = Hs_to_Ds_old_data(zs_modelo,H_modelo,z_data,i)
        chies[i] = chi_2_BAO(outs,valores_data_2,errores_data_2)
    if np.isnan(sum(chies))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)

    return np.sum(chies)

if __name__ == '__main__':
    from scipy.constants import c as c_luz #metros/segundos

    from matplotlib import pyplot as plt
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    #%% BAO
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_viejos')
    archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']

    def leer_data_BAO(archivo_BAO):
            z, valores_data, errores_data, rd_fid = np.loadtxt(archivo_BAO,
            usecols=(0,1,2,4),unpack=True)
            return z, valores_data, errores_data, rd_fid

    dataset_BAO = []
    for i in range(5):
        aux = leer_data_BAO(archivo_BAO[i])
        dataset_BAO.append(aux)
#%%
    num_datasets = 5
    #[omega_m,H_0] = [0.33013649504023296, 66.48702802652504]
    [omega_m,H_0] = [0.298,73.5]
    zs_modelo = np.linspace(0.01, 3, 10**6)
    H_modelo = H_LCDM(zs_modelo, omega_m, H_0)
    rd = r_drag_camb(omega_m,H_0) #Calculo del rd

    legends = ['Da','Dh','Dm','Dv','H']
    chies = np.zeros(num_datasets)
    for i in range(5):
        (z_data, valores_data, errores_data, rd_fid) = dataset_BAO[i]
        if i==0: #Dato de Da
            valores_data_2 = valores_data * (rd/rd_fid)
            errores_data_2 = errores_data * (rd/rd_fid)
        elif i==4: #Datos de H
            valores_data_2 = np.zeros(len(valores_data))
            errores_data_2 = np.zeros(len(errores_data))
            for j in range(len(z_data)):
                valores_data_2[j] = valores_data[j] * (rd_fid[j]/rd)
                errores_data_2[j] = errores_data[j] * (rd_fid[j]/rd)
        else:
            valores_data_2 = np.zeros(len(valores_data))
            errores_data_2 = np.zeros(len(errores_data))
            for j in range(len(z_data)):
                if rd_fid[j] != 1:
                    valores_data_2[j] = valores_data[j] * (rd/rd_fid[j])
                    errores_data_2[j] = errores_data[j] * (rd/rd_fid[j])
                else: #No hay que multiplicar x ningun factor
                    valores_data_2[j] = valores_data[j]
                    errores_data_2[j] = errores_data[j]
        outs = Hs_to_Ds_old_data(zs_modelo,H_modelo,z_data,i)
        chies[i] = chi_2_BAO(outs,valores_data_2,errores_data_2)
    print(np.sum(chies)/(17-2))

    plt.close()
    plt.figure(2)
    plt.grid()
    plt.errorbar(z_data,valores_data_2,errores_data_2,fmt='.r')
    plt.plot(z_data,outs,'.-b')#los puntos unidos por lineas, no es la forma real
    plt.title(legends[i])
    plt.xlabel('z (redshift)')
    plt.ylabel(legends[i])
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_viejos/Imagen')
    #os.chdir(path_git+'/Software/Estadística/Datos/BAO/Datos_sin_nuevos/Imagen')
    #plt.savefig('{}'.format(legends[i]))
