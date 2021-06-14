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

def r_drag(omega_m,H_0,z_i=1100,z_f=3600):
    #Calculo del rd:
    h = H_0/100
    omega_b = 0.05 #(o algo asi)
    cte = 0.02 #(omega_B * h**2)

    #Calculo el zd
    b1 = 0.313*(omega_m*h**2)**(-0.419)*(1+0.607*(omega_m*h**2)**(0.6748))
    b2 = 0.238*(omega_m*h**2)**0.223
    zd = ((1291*(omega_m*h**2)**0.251)/(1+0.659*(omega_m*h**2)**0.828)) * (1+b1*cte**b2)

    zs_int = np.logspace(np.log10(zd),13,int(10**5))
    H_int = H_LCDM(zs_int,omega_m,H_0)

    R_bar = 31500 * cte * (2.726/2.7)**(-4)
    #cs = c_luz_km / (np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    #integrando = cs/H_int
    integrando = c_luz_km / (H_int * np.sqrt(3*(1 + R_bar/(1 + zs_int))))
    rd = simps(integrando,zs_int)
    return rd

def chi_2_BAO(teo, data, errores):
    chi2 = np.sum(((data-teo)/errores)**2)
    return chi2

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

def params_to_chi2_BAO_AGN(theta, params_fijos, dataset, z_data,
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

    #Para BAO
    zs = np.linspace(0.01, 3, cantidad_zs) #Sirve para interpolar los D_i
    H_teo = H_LCDM(zs, omega_m, H_0)
    rd = r_drag(omega_m,H_0)

    #Para AGN
    zs_modelo = np.linspace(0,30,cantidad_zs)
    H_modelo = H_LCDM(zs_modelo, omega_m, H_0)

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

        outs = Hs_to_Ds(zs,H_teo,z_data_BAO,i)
        chies_BAO[i] = chi_2_BAO(outs,valores_data_mod,errores_data_mod)
        #chies_BAO[i] = chi_2_BAO(outs,valores_data,errores_data) #esto esta mal pero es como lo hiciste en la tesis!

    if np.isnan(sum(chies_BAO))==True:
        print('Hay errores!')
        print(omega_m,H_0,rd)
    chi2_BAO = np.sum(chies_BAO)

    #AGN
    Da = Hs_to_Da(zs_modelo,H_modelo,z_data)
    Theta_theo = theta_teorico(z_data,Da,Sobs,alpha,l_fijo=True)
    chi2_AGN = chi_2_AGN(Theta_theo,Theta_data,np.sqrt(1.01)*dTheta)

    return chi2_BAO + chi2_AGN

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
    #Data AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')

    #z_data, Theta_data, dTheta,Sobs,dSobs, alpha = leer_data_AGN('datosagn_less.dat',less_data=True)

    z_data, Theta_data, dTheta,Sobs, alpha = leer_data_AGN('datosagn.dat',less_data=False)
    dSobs = np.zeros(len(z_data))

    H_0 = 73.48
    params_fijos = H_0
    omegas = np.linspace(0.1,1,50)
    omegas

    output = np.zeros(len(omegas))
    for i,omega in enumerate(omegas):
        theta = omega
        output[i] = params_to_chi2_BAO_AGN(theta, params_fijos, dataset,
                                z_data,Theta_data,dTheta,
                                Sobs,dSobs,alpha)
    %matplotlib qt5
    plt.plot(omegas,output)
    index = np.where(output==min(output))[0][0]
    omegas[index]
    min(output/(17+len(z_data)-1))
