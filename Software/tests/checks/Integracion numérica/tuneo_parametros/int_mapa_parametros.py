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
c_luz_km = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from condiciones_iniciales import condiciones_iniciales
from cambio_parametros import params_fisicos_to_modelo
from int import dX_dz


#%%
sistema_ec=dX_dz
z_inicial = 30
z_final = 0
cantidad_zs = int(10**5)
max_step = 0.003
n=1

zs = np.linspace(z_inicial,z_final,cantidad_zs)

omegas = np.linspace(0.32,0.41,20)
bs = np.linspace(0.2,0.3,50)
errores = np.zeros((len(omegas),len(bs)))
t1 = time.time()
for i, omega_m in enumerate(omegas):
    for j, b in enumerate(bs):

        cond_iniciales = condiciones_iniciales(omega_m,b)
        eta = (c_luz_km/(8315*100)) * np.sqrt(omega_m/0.13)
        c1, c2 = params_fisicos_to_modelo(omega_m,b)

        #Integramos el sistema
        zs_int = np.linspace(z_inicial,z_final,cantidad_zs)
        sol = solve_ivp(sistema_ec, (z_inicial,z_final),
            cond_iniciales, t_eval=zs_int, args=[c1,c2,n], max_step=max_step)

        #if (sol.success!=True): #Otra formaa

        if (len(sol.t)!=cantidad_zs):
            errores[i,j] = 3 #Esto es lo peor
        elif np.all(zs==sol.t)==False:
            errores[i,j] = 2 #Esto es menos peor
        else:
            errores[i,j] = 1 #Esto es correcto :)
        #dos y tres son errores, uno es todo piola
t2 = time.time()
print('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
      int((t2-t1) - 60*int((t2-t1)/60))))

om_malos = omegas[np.where(errores==3)[0]]
bs_malos = bs[np.where(errores==3)[1]]
print(om_malos)
print(bs_malos)

#%%
%matplotlib qt5
plt.close()
plt.figure(1)
plt.xlabel('b')
plt.ylabel('omega_m')
plt.scatter(bs_malos,om_malos)
plt.figure(2)
plt.matshow(errores)
#plt.colorbar()
plt.show()
errores


#Errores que aparecieron en la integración para SN:
#[0.35555556, 0.11034483]
#[0.36666667, 0.15517241]
#[0.35263158, 0.10461538]
#[0.33351188   0.11611847]*
#[0.36000551   0.10510931]*
#(*) errores que aparecieron con los nuicense aprameters de SN

#Barrido exhautivo para  entre 0.1 y 0.2 y omega entre 0.3 y 0.4:
#[0.34210526 0.35789474 0.36315789 0.36842105 0.38421053 0.38947368
# 0.39473684 0.39473684 0.39473684 0.39473684 0.39473684]
#[0.10122449 0.10653061 0.10285714 0.1155102  0.10489796 0.10285714
# 0.1044898  0.10530612 0.10571429 0.1077551  0.10897959]

#[0.35263158 0.35894737 0.36210526 0.37157895 0.37473684 0.37473684
# 0.37473684 0.37789474 0.38105263 0.38421053 0.39052632 0.39052632
# 0.39052632 0.39368421 0.39684211 0.39684211 0.4       ]
#[0.1        0.10122449 0.1        0.13918367 0.10489796 0.10612245
# 0.15632653 0.12938776 0.10857143 0.10489796 0.10489796 0.12571429
# 0.13795918 0.13306122 0.10489796 0.13183673 0.10734694]


#[0.41578947 0.44736842 0.45789474 0.45789474 0.46842105 0.47894737
# 0.48947368]
#[0.2122449  0.25020408 0.2122449  0.24285714 0.22938776 0.22571429
# 0.22571429]
