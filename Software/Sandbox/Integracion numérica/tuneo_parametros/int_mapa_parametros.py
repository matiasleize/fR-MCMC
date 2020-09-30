"""
Created on Sun Feb  2 13:28:48 2020

@author: matias

En este script se realiza el mapa de parámetros y se grafica la zona donde la
integración numérica funciona

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import time
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_condiciones_iniciales import condiciones_iniciales
from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_int import dX_dz


#%%
sistema_ec=dX_dz
z_inicial = 0
z_final = 3
cantidad_zs = 1000000
max_step = np.inf #0.01
verbose = True
H0=73.48
n=1

zs = np.linspace(z_inicial,z_final,cantidad_zs)

omegas = np.linspace(0.01,0.8,10)
bs = np.linspace(-0.01,1.5,10)

errores = np.zeros((len(omegas),len(bs)))
for i, omega_m in enumerate(omegas):
    for j, b in enumerate(bs):

        params_fisicos = [omega_m,b,H0]

        #Calculo las condiciones cond_iniciales, eta
        # y los parametros de la ecuación
        cond_iniciales = condiciones_iniciales(*params_fisicos)
        eta = (c_luz_norm/(8315*100)) * np.sqrt(omega_m/0.13)
        c1, c2 = params_fisicos_to_modelo(omega_m,b)

        #Integramos el sistema
        zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
        sol = solve_ivp(sistema_ec, (z_inicial,z_final),
            cond_iniciales, t_eval=zs_int, args=[c1,c2,n], max_step=max_step)

        #if (sol.success!=True): #Otra formaa
        if (len(sol.t)!=cantidad_zs):
            errores[i,j] = 1
        if np.all(zs==sol.t)==False:
            errores[i,j] = 2

%matplotlib qt5
#np.where(errores==1)
om_malos = omegas[np.where(errores==1)[0]]
bs_malos = bs[np.where(errores==1)[1]]
#print(bs_malos)
plt.close()
plt.figure(1)
plt.xlabel('b')
plt.ylabel('omega_m')
plt.scatter(bs_malos,om_malos)
plt.figure(2)
plt.matshow(errores)
#plt.colorbar()
plt.show()
