"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import time
import camb
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
from funciones_AGN import zs_2_logDlH0

#%%
'''
DEPRECATED: Antes de eliminar este archivo copiar este ejemplo en otro .py
en donde se grafiquen los datos.
'''
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
    from funciones_data import leer_data_AGN

    #%%

    def leer_data_AGN(archivo_AGN):
        z, Fuv, eFuv, Fx, eFx = np.loadtxt(archivo_AGN,
        usecols=(3,4,5,6,7), unpack=True)
        arr1inds = z.argsort()
        sorted_z = z[arr1inds]
        sorted_Fuv = Fuv[arr1inds]
        sorted_eFuv = eFuv[arr1inds]
        sorted_Fx = Fx[arr1inds]
        sorted_eFx = eFx[arr1inds]
        return sorted_z, sorted_Fuv, sorted_eFuv, sorted_Fx, sorted_eFx

    #Data AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
    data_agn = leer_data_AGN('table3.dat')


    H_0 = 70
    omega_m = 0.99
    gamma = 0.64
    beta = 7
    delta = 0.3
    theta = [omega_m,beta,gamma,delta]

    #params_to_chi2_AGN_nuisance(theta, _, data_agn)/(len(z_data)-4)

    data_agn = leer_data_AGN('table3.dat')
    z_data_1, logFuv_1, eFuv_1, logFx_1, eFx_1  = data_agn

    zmin = 0
    zmax = 100
    mask = (z_data_1 > zmin) & (z_data_1 < zmax)

    z_data = z_data_1[mask]
    logFuv = logFuv_1[mask]
    logFx = logFx_1[mask]
    eFx = eFx_1[mask]
    eFuv = eFuv_1[mask]

    zs_modelo = np.linspace(0,30,10**6)
    Dl_teo = -np.log10(H_0) + zs_2_logDlH0(zs_modelo,omega_m,z_data)
    Dl_teo_cm = Dl_teo - np.log10(3.24) + 25
    psi = beta + gamma * logFuv + 2 * (gamma-1) * (Dl_teo_cm + 0.5 * np.log10(4*np.pi))

    si_2 = eFx**2 + (gamma * eFuv)**2 + np.exp(2*np.log(delta)) #El cuadrado de los errores
    #si_2 = eFx**2 + (gamma * eFuv)**2 + delta**2 #El cuadrado de los errores
    print(np.sum(si_2))
    chi2_AGN = np.sum( ((logFx-psi)**2/si_2) + np.log(2*np.pi*si_2)) # menos en el paper

    print(chi2_AGN)
    print(chi2_AGN/(len(z_data)-4))

    plt.figure()
    plt.xlabel('z (redshift)')
    plt.ylabel(r'$Fx$')
    plt.errorbar(z_data,psi,np.sqrt(si_2),marker='.',linestyle='')
    plt.plot(z_data,logFx,'.r')
