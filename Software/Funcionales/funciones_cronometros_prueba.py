"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""
import numpy as np
import sys
import os
from os.path import join as osjoin
from scipy.interpolate import interp1d

from scipy.constants import c as c_luz #metros/segundos
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo
from HS_taylor import Taylor_HS


#ORDEN DE PRESENTACION DE LOS PARAMETROS: omega_m,b,H_0,n

def chi_2_cronometros(H_teo, H_data, dH):
    chi2 = np.sum(((H_data - H_teo) / dH)**2)
    return chi2

def params_to_chi2_cronometros(theta, z_data, H_data,
                                dH, cantidad_zs=int(10**5)):
    '''Dados los parámetros libres del modelo (omega y H0), devuelve un chi2
     para los datos de los cronómetros cósmicos'''

    [omega_m, H_0] = theta
    z = np.linspace(0, 3, cantidad_zs)
    H = H_LCDM(z, omega_m, H_0)
    H_int = interp1d(z, H)
    H_teo = H_int(z_data)
    chi = chi_2_cronometros(H_teo, H_data, dH)
    return chi


def params_to_chi2_taylor(theta, params_fijos, z_data, H_data, dH,cantidad_zs=int(10**5)):
    '''Dados los parámetros libres del modelo (omega, b y H0) y los que quedan params_fijos (n),
    devuelve un chi2 para los datos de cronómetros cósmicos'''

    [omega_m, b] = theta
    [H_0,n] = params_fijos

    zs = np.linspace(0,3,cantidad_zs)
    H_taylor = Taylor_HS(zs, omega_m, b, H_0)
    H_int = interp1d(zs, H_taylor)
    H_teo = H_int(z_data)

    chi = chi_2_cronometros(H_teo,H_data,dH)
    #return chi/(len(z_data)-len(theta))
    return chi

#%%
import sympy as sym
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_cronometros
os.chdir(path_git+'/Software/Estadística/Datos/')
z_data, H_data, dH  = leer_data_cronometros('datos_cronometros.txt')



def H_LCDM(z, omega_m, H_0):
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H



#Parámetros
from matplotlib import pyplot as plt
%matplotlib qt5
omega_m = 0.5
b = 2
H0 = 73.48
n=1



zs = np.linspace(0,2,10000);
H_LCDM = H_LCDM(z_data,omega_m,H0)
H_taylor = Taylor_HS(z_data,omega_m,b,H0)

plt.plot(z_data,H_LCDM,'bo')
plt.plot(z_data,H_taylor,'r.')
plt.plot(z_data,H_data,'g.')
plt.errorbar(z_data,H_data,dH)
plt.hlines(0, xmin =0 ,xmax = 2)
plt.show()

chi = chi_2_cronometros(H_taylor, H_data, dH)
chi_norm = chi/(len(z_data)-2)
print(chi_norm)


#%%
bs = np.linspace(-5,5,10)
omegas = np.linspace(0.08,0.8,20)

chies=np.zeros((len(omegas),len(bs)))
for i,omega in enumerate(omegas):
    for j, b0 in enumerate(bs):
        #H_taylor = Taylor_HS(z_data,omega,b0,H0)
        #chies[i,j] = chi_2_cronometros(H_taylor, H_data, dH)
        chies[i,j] = params_to_chi2_taylor([omega,b0], [H0,n], z_data, H_data, dH)

plt.close()
plt.matshow(np.exp(-0.5*chies))
plt.matshow(chies)
plt.colorbar()
plt.show()
