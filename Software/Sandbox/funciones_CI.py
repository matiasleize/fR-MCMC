"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales/')

from funciones_int import integrador
from funciones_taylor import Taylor_HS, Taylor_ST

from scipy.integrate import cumtrapz as cumtrapz
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_km = c_luz/1000 #km/seg
#ORDEN DE PRESENTACION DE LOS PARAMETROS: Mabs,omega_m,b,H_0,n

z_final = 30
cantidad_zs = int(10**5)
max_step = 10**(-5)

b = 0.5
omega_m = 0.24
H0 = 73.48
params_fisicos = [omega_m,b,H0]

zs_ref, H_ref = integrador(params_fisicos, n=2, cantidad_zs=cantidad_zs,
            max_step=max_step, z_inicial=z_final, z_final=0,
            model='HS')

f = interp1d(zs_ref,H_ref)


z_inicial = np.linspace(3,z_final,5)
error_porcentual = np.zeros(len(z_inicial))
for i,z0 in enumerate(z_inicial):

    zs, H_ode = integrador(params_fisicos, n=2, cantidad_zs=cantidad_zs,
                max_step=max_step, z_inicial=z0, z_final=0,
                model='HS')
    H_ref = f(zs)

    error_porcentual[i] = 100*np.abs(np.mean(1-(H_ode/H_ref)))
#%%

from matplotlib import pyplot as plt
plt.plot(z_inicial,error_porcentual,label='HS')
plt.Title('Error porcentual cometido al achicar la CI')
plt.xlabel('z0 (redshift)')
plt.ylabel('Error porcentual')
plt.grid(True)
plt.legend(loc='best')
