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
from scipy.integrate import simps as simps
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000
#%%

def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

#%% AGN + BAO
def Hs_to_Da(zs, Hs, z_data):
    INT = cumtrapz(Hs**(-1), zs, initial=0)
    DA = (c_luz_km/(1 + zs)) * INT

    output = interp1d(zs,DA)
    return output(z_data)

def chi_2_AGN(teo, data, errores):
    #1 miliarcsec = 4.8481368110954**(-9) rad
    teo_miliarcsec = teo * 206264806.24709466 #Paso la \Theta_{teo} de rad a miliarcsec
    chi2 = np.sum(((data-teo_miliarcsec)/errores)**2)
    return chi2

def theta_teorico(zs,Da,Sobs,alpha,l=11.03,n_agn=0,beta=0,l_fijo=False):

    # L = 4 * np.pi * Sobs * (D_a)**2 / (1+z)**(alpha-3)
    # gamma = ((L/L0)**beta) /D_a
    #Theta = l * gamma * (1+z)**n

    if beta==0:
        if n_agn==0:
            if l_fijo==True:
                    Theta_theo = 11.03 / (Da * 10**6) #Pasamos Da de Mpc a pc así theta es adim (en rad)
            else:
                Theta_theo = l / (Da * 10**6) #Pasamos Da de Mpc a pc así theta es adim (en rad)
        else:
            Theta_theo = (l * (1+zs)**n_agn) / (Da*10**(6))
    else:
        L_0 = 10**(28)
        delta = 1/30856778200000000 #Convertion meters to pc
        Theta_theo = l * (4 * np.pi * Sobs/(L_0*(delta**2)))**beta * (Da*10**6)**(2*beta-1) * (1+zs)**(n_agn-beta*(alpha-3))
    return Theta_theo

def params_to_chi2_AGN(theta, params_fijos, z_data,
                            Theta_data,dTheta,Sobs,dSobs,alpha,
                            num_datasets=5, cantidad_zs=10**5,
                            l_fijo=True):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''
#    if len(theta)==2:
#        [omega_m, H_0] = theta
    #else: #Si theta tiene dim 1
    omega_m = theta
    H_0 = params_fijos

    #Para AGN
    zs_modelo = np.linspace(0,30,cantidad_zs)
    H_modelo = H_LCDM(zs_modelo, omega_m, H_0)

    Da = Hs_to_Da(zs_modelo,H_modelo,z_data)
    Theta_theo = theta_teorico(z_data,Da,Sobs,alpha,l_fijo=True)
    chi2_AGN = chi_2_AGN(Theta_theo,Theta_data,np.sqrt(1.01)*dTheta)

    return chi2_AGN

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

    #Data AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')

    #z_data, Theta_data, dTheta,Sobs,dSobs, alpha = leer_data_AGN('datosagn_less.dat',less_data=True)
    z_data, Theta_data, dTheta,Sobs, alpha = leer_data_AGN('datosagn.dat',less_data=False)
    dSobs = np.zeros(len(z_data))

    H_0 = 73.48
    params_fijos = H_0
    omegas = np.linspace(0.1,1,50)
    output = np.zeros(len(omegas))
    for i,omega in enumerate(omegas):
        theta = omega
        output[i] = params_to_chi2_AGN(theta, params_fijos,z_data,Theta_data,dTheta,Sobs,dSobs,alpha)
    #%matplotlib qt5
    plt.plot(omegas,output)
    index = np.where(output==min(output))[0][0]
    omegas[index]
    min(output/(len(z_data)-1))

    #%%
    omega_m = omegas[index]  #Best fit
    l = 11.03 #En pc
    cantidad_zs = 10**5
    theta = omega_m
    params_fijos = H_0

    zs_modelo = np.linspace(0,30,cantidad_zs)
    H_modelo = H_LCDM(zs_modelo, omega_m, H_0)

    DA = Hs_to_Da(zs_modelo,H_modelo,z_data)
    DA_pc =  DA * 10**6 #en pc

    Theta_teo = 206264806.24709466*l/DA_pc #En miliarcseg

    Delta_Theta = Theta_data - Theta_teo
    chi2 = np.sum((Delta_Theta/dTheta)**2)
    chi2/(len(z_data)-1)

    #%%
    %matplotlib qt5
    #plt.figure(1)
    plt.title('$\Lambda CDM$ ($H_{0}$=73.48 $\\frac{km/seg}{Mpc}$)')
    plt.errorbar(z_data, Theta_data, yerr=dTheta,fmt='.',label='$\\theta_{data}$')
    plt.plot(z_data, Theta_teo,'-',label='$\\theta_{Teo}$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('$\Theta$ (miliarcsec)')
