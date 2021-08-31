"""
Created on Wed Feb  5 13:04:17 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_data import leer_data_pantheon
from funciones_LambdaCDM_1 import params_to_chi2
#%%

os.chdir(path_git+'/Software/Estadística/Datos/Datos_pantheon/')
zcmb,zhel, Cinv, mb = leer_data_pantheon('lcparam_full_long_zhel.txt')
sn = len(zcmb)
Cinv.shape

#%%
z = np.linspace(0, 3, int(10**6))

def H_LCDM(z, omega_m, H_0):
    '''Dado un array de z's, y ciertos parametros omega_m y H0, devuelve
    un array de H's correspondiente a LCDM.'''
    omega_lambda = 1 - omega_m
    H = H_0 * np.sqrt(omega_m * (1 + z)**3 + omega_lambda)
    return H

def distancia_luminosa(z):
    H = H_LCDM(z,omega_matter,H0)
    d_c =  c_luz_km * cumtrapz(H**(-1), z, initial=0)  #Integro 1/H(z)
    d_L = (1 + z) * d_c
    return d_L #Devuelve un array de longitud 10**6!

def distancia_luminosa_data(z,zcmb,zhel):
    H = H_LCDM(z,omega_matter,H0)
    d_c =  c_luz_km * cumtrapz(H**(-1), z, initial=0) #Integro 1/H(z)
    dc_int = interp1d(z, d_c) #Interpolamos
    d_L = (1 + zhel) * dc_int(zcmb)
    return d_L #Devuelve un array de longitud 1048.
#%%
#Parametros a ajustar
name = 'data_16'
omega_matter = 0.8
H0 =  60 #Unidades de (km/seg)/Mpc
#%%
os.chdir(path_datos_global+'/Para Lucila/muchos_z')
d_L = distancia_luminosa(z)
file2write=open(name,'w+')
for i in range(len(z)):
    file2write.write( '{} \t {}\n'.format(z[i], d_L[i]))
file2write.close()

#%%
os.chdir(path_datos_global+'/Para Lucila/pocos_z')
d_L_data = distancia_luminosa_data(z,zcmb,zhel)
file2write=open(name,'w+')
for i in range(len(zcmb)):
    file2write.write( '{} \t {}\n'.format(zcmb[i], d_L_data[i]))
file2write.close()
