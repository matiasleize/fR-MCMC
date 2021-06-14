"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from camb import model, initialpower
import time
import scipy
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as cumtrapz
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
#%%

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

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

def r_drag_camb(omega_m,H_0):
    pars = camb.CAMBparams()
    omega_bh2 = 0.0225 #BBN
    pars.set_cosmology(H0=H_0, ombh2=omega_bh2, omch2=omega_m*(H_0/100)**2-omega_bh2)
    results = camb.get_background(pars)
    rd = results.get_derived_params()['rdrag']
    #print('Derived parameter dictionary: %s'%results.get_derived_params()['rdrag'])
    return rd


def chi_2_BAO(teo, data, errores):
    chi2 = np.sum(((data-teo)/errores)**2)
    return chi2

def params_to_chi2_BAO(theta, params_fijos, dataset_BAO,
                        cantidad_zs=int(10**6),num_datasets=5):
    '''Dados los parámetros libres del modelo (omega, b y H0) y
    los que quedan params_fijos (n), devuelve un chi2 para los datos
    de BAO'''

    [omega_m, H_0] = theta
    zs = np.linspace(0.01, 3, cantidad_zs)
    H_teo = H_LCDM(zs, omega_m, H_0)
    rd = r_drag_camb(omega_m,H_0) #Calculo del rd

    #La parte obs del chi2
    chies_BAO = np.zeros(num_datasets)
    for i in range(num_datasets):
        (z_data_BAO, valores_data, errores_data, rd_fid) = dataset[i]
        if isinstance(z_data_BAO,np.float64):
            if (rd_fid == 1): #Abreviatura de que el cociente rd/rd_fid=1
                valores_data_mod = valores_data
                errores_data_mod = errores_data
            else:
                valores_data_mod = valores_data * (rd/rd_fid)
                errores_data_mod = errores_data * (rd/rd_fid)
        elif isinstance(z_data_BAO,np.ndarray):
            valores_data_mod=np.zeros(len(valores_data))
            errores_data_mod=np.zeros(len(errores_data))
            for j in range(len(z_data_BAO)):
                if (rd_fid[j] == 1):
                        valores_data_mod[j] = valores_data[j]
                        errores_data_mod[j] = errores_data[j]
                else:
                    valores_data_mod[j] = valores_data[j] * (rd/rd_fid[j])
                    errores_data_mod[j] = errores_data[j] * (rd/rd_fid[j])
        #La parte teo del chi2
        outs = Hs_to_Ds(zs,H_teo,z_data_BAO,i)

        #Calculo el chi2 para cada tipo de dato (i)

        chies_BAO[i] = chi_2_BAO(outs,valores_data_mod,errores_data_mod)
#        chies_BAO[i] = chi_2_BAO(outs,valores_data,errores_data) #esto esta mal pero es como lo hiciste en la tesis!
    if np.isnan(sum(chies_BAO))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)

    return np.sum(chies_BAO)
#%%

if __name__ == '__main__':
    from scipy.constants import c as c_luz #metros/segundos

    from matplotlib import pyplot as plt
    import sys
    import os
    from os.path import join as osjoin
    from pc_path import definir_path
    path_git, path_datos_global = definir_path()
    os.chdir(path_git)
    sys.path.append('./Software/Funcionales/')
    from funciones_data import leer_data_BAO,leer_data_AGN

    #Data BAO
    os.chdir(path_git+'/Software/Estadística/Datos/BAO/')
    dataset = []
    archivo_BAO = ['datos_BAO_da.txt','datos_BAO_dh.txt','datos_BAO_dm.txt',
                    'datos_BAO_dv.txt','datos_BAO_H.txt']
    for i in range(5):
        aux = leer_data_BAO(archivo_BAO[i])
        dataset.append(aux)


    H_0 = 73.48
    H_01 = H_0
    H_02 = 65
    params_fijos = H_0
    omegas = np.linspace(0.1,1,50)
    output = np.zeros(len(omegas))
    rds1 = np.zeros(len(omegas))
    rds2 = np.zeros(len(omegas))
    for i,omega in enumerate(omegas):
        theta = [omega,H_0]
        output[i] = params_to_chi2_BAO(theta, params_fijos, dataset)
        rds1[i]= r_drag_camb(omega,H_01)
        rds2[i]= r_drag_camb(omega,H_02)
    %matplotlib qt5
    plt.plot(omegas,output)
    index = np.where(output==min(output))[0][0]
    omegas[index]
    min(output/(17-1))

    plt.grid(True)
    plt.xlabel(r'$\Omega_{m}$')
    plt.ylabel('rd(Mpc)')
    plt.plot(omegas,rds1,label='H0=73.48')
    plt.plot(omegas,rds2,label='H0=65')
    plt.legend(loc='best')
