"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz as cumtrapz

from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000;

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_data import leer_data_AGN
from funciones_taylor import Taylor_HS, Taylor_ST

#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

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

def theta_teorico(zs,Da,Sobs,alpha,l,n_agn,beta,l_fijo=False):

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

def params_to_chi2(theta, params_fijos, z_data,Theta_data,dTheta,Sobs,dSobs,alpha,
                    verbose=True,cantidad_zs=10**5,model='HS',M_fijo=False,
                    chi_riess=True,taylor=False,l_fijo=True):
    '''Dados los parámetros del modelo devuelve un chi2 para los datos
    de supernovas. 1 parámetro fijo y 4 variables'''

    if len(theta)==2:
            [omega_m,b] = theta
            [H_0,l,n,beta,n_agn] = params_fijos

    elif len(theta)==3:
            [omega_m,b,l] = theta
            [H_0,n,beta,n_agn] = params_fijos

    elif len(theta)==4:
        [omega_m,b,H_0,l] = theta
        [n,beta,n_agn] = params_fijos

    elif len(theta)==5:
        [omega_m,b,l,beta,n_agn] = theta
        [H_0,n] = params_fijos

    elif len(theta)==6:
        [omega_m,b,H_0,l,beta,n_agn] = theta
        n = params_fijos

    if taylor == True:
        zs_modelo = np.linspace(0,30,cantidad_zs)
        if model=='HS':
            #H_modelo_cron = Taylor_HS(z_data_cron, omega_m, b, H_0)
            H_modelo = Taylor_HS(zs_modelo, omega_m, b, H_0)
        else:
            #H_modelo_cron = Taylor_ST(z_data_cron, omega_m, b, H_0)
            H_modelo = Taylor_ST(zs_modelo, omega_m, b, H_0)

    else:
        if (-np.inf < b < 0.2):
            zs_modelo = np.linspace(0,30,cantidad_zs)
            if model=='HS':
                #H_modelo_cron = Taylor_HS(z_data_cron, omega_m, b, H_0)
                H_modelo = Taylor_HS(zs_modelo, omega_m, b, H_0)
            else:
                #H_modelo_cron = Taylor_ST(z_data_cron, omega_m, b, H_0)
                H_modelo = Taylor_ST(zs_modelo, omega_m, b, H_0)
        else:
            params_fisicos = [omega_m,b,H_0]
            zs_modelo, H_modelo = integrador(params_fisicos, n, model=model)
            #Para SN se interpola con datos en el calculo de mu!

        #_int = interp1d(zs_modelo, H_modelo)
        #H_modelo_cron = H_int(z_data_cron)
    Da = Hs_to_Da(zs_modelo,H_modelo,z_data)
    Theta_theo = theta_teorico(z_data,Da,Sobs,alpha,l,n_agn,beta,l_fijo=l_fijo)
    chi2 = chi_2_AGN(Theta_theo,Theta_data,np.sqrt(1.01)*dTheta)
    #print(Theta_theo/4.8481368110954**(-9),Theta_data)
    return chi2

#%%
if __name__ == '__main__':
    from scipy.constants import c as c_luz #metros/segundos
    from funciones_data import leer_data_AGN
    os.chdir(path_git+'/Software/Estadística/Datos/Datos_AGN')
    z_data, Theta_data, dTheta,Sobs,_ , alpha = leer_data_AGN('datosagn_less.dat')
    dSobs = np.zeros(len(z_data))
    n = 0
    beta = 0
    n_agn = 0

    omega_m = 0.222
    b = 0.1
    H_0 = 73.48
    l = 13.01#En pc

    cantidad_zs = 10**5
    model = 'HS'

    theta = [omega_m,b,H_0,l]
    params_fijos = [n,beta,n_agn]

    if len(theta)==3:
            [omega_m,b,l] = theta
            [H_0,n,beta,n_agn] = params_fijos

    elif len(theta)==4:
        [omega_m,b,H_0,l] = theta
        [n,beta,n_agn] = params_fijos

    elif len(theta)==5:
        [omega_m,b,l,beta,n_agn] = theta
        [H_0,n] = params_fijos

    elif len(theta)==6:
        [omega_m,b,H_0,l,beta,n_agn] = theta
        n = params_fijos

    if (-np.inf < b < 0.2):
        zs_modelo = np.linspace(0,30,cantidad_zs)
        if model=='HS':
            H_modelo = Taylor_HS(zs_modelo, omega_m, b, H_0)
        else:
            H_modelo = Taylor_ST(zs_modelo, omega_m, b, H_0)
    else:
        params_fisicos = [omega_m,b,H_0]
        zs_modelo, H_modelo = integrador(params_fisicos, n, model=model)

    DA = Hs_to_Da(zs_modelo,H_modelo,z_data)
    DA_pc =  DA * 10**6 #en pc

    Theta_teo = 206264806.24709466*l/DA_pc #En miliarcseg
#    dTheta_1 = dTheta*6

    Delta_Theta = Theta_data - Theta_teo
    chi2 = np.sum((Delta_Theta/dTheta)**2)
    chi2/(len(z_data)-4)
    chi2

    #%%
    import pylab
    import scipy.stats as stats
    from matplotlib import pyplot as plt
    %matplotlib qt5

    plt.figure(1)
    plt.title('Theta vs z')
    plt.errorbar(z_data, Theta_data, yerr=dTheta,fmt='.',label='$\\theta_{data}$')
    plt.plot(z_data, Theta_teo,'-',label='$\\theta_{Teo}$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('z (redshift)')
    plt.ylabel('$\Theta$ (miliarcsec)')
    z_data
    #%%
    Chi2 = Delta_Theta/dTheta
    #Shapiro-wilk test for gaussianity
    stats.shapiro(Chi2)
    #&&
    plt.figure(2)
    stats.probplot(Chi2, dist="norm", plot=pylab); pylab.show()
    #%%
    plt.figure(3)
    fig, ax = plt.subplots(figsize=[8,6])
    ax.set_title("Some title")
    ax.set_xlabel("z (redshift)")
    ax.set_ylabel(r"$\Delta\Theta / \sigma$")
    N, bins, patches = ax.hist(Chi2, bins=int(np.sqrt(len(Chi2))), color="#777777") #initial color of all bin
    plt.show()
