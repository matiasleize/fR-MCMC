"""
Created on Sun Feb  2 13:28:48 2020

@author: matias

En este script se realiza el mapa de parámetros y se grafica la zona donde la
integración numérica funciona

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz
import time

from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000

def params_fisicos_to_modelo(omega_m, b, n=1):
    '''Toma los parametros fisicos (omega_m, el parametro de distorsión b)
    y devuelve los parametros del modelo c1 y c2'''
    c_luz_norm = c_luz/1000 #km/seg
    alpha = 1 / (8315)**2
    beta = 1 / 0.13
    aux = ((100/c_luz_norm)**2 * 6 * (1 - omega_m))  / (alpha * omega_m * beta)
    c_2 =  (2/(aux * b))
    c_1 =  2/b
    return c_1, c_2


def dX_dz(z, variables,*params_modelo, model='HS'):
    '''Defino el sistema de ecuaciones a resolver. El argumento params_modelo
    es una lista donde los primeros n-1 elementos son los parametros del sistema,
    mientras que el útimo argumento especifica el modelo en cuestión,
    matematicamente dado por la función gamma.'''

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]
    r = variables[4]

    [B,D,N] = params_modelo
    gamma = lambda r,b,d: ((1+d*r) * (-b*r + r*(1+d*r)**2)) / (b*2*d*r**2)
    G = gamma(r,B,D)

#    gamma = lambda r,b,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*(1 - n + d * (1+n) * r**n)*r**n)
#    G = gamma(r,B,D,N)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = (- (v*x*G - x*y + 4*y - 2*y*v)) / (z+1)
    s2 = (-v * (x*G + 4 - 2*v)) / (z+1)
    s3 = (w * (-1 + x + 2*v)) / (z+1)
    s4 = (-x * r * G) / (1+z)
    return [s0,s1,s2,s3,s4]
#%%
sistema_ec=dX_dz
z_inicial = 0
z_final = 3
cantidad_zs = 1000000
max_step = np.inf
verbose = True

x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales = ci
#%%

zs = np.linspace(z_inicial,z_final,cantidad_zs)

omegas = np.linspace(0.01,0.8,60)
bs = np.linspace(0.01,1.5,60)

errores = np.zeros((len(omegas),len(bs)))
for i, omega in enumerate(omegas):
    for j, b in enumerate(bs):
        c1,c2 = params_fisicos_to_modelo(omega,b)
        params_modelo=[c1, c2, 1]

        # Integ ramos el vector v y calculamos el Hubble
        sol = solve_ivp(sistema_ec, (z_inicial,z_final),cond_iniciales, t_eval=zs,
                        args=params_modelo)#,max_step=0.001)

        if (sol.success!=True): #Otra formaa        
        #if (len(sol.t)!=cantidad_zs):
            errores[i,j] = 1
#        if np.all(zs==sol.t)==False:
#            errores[i,j] = 2

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


#Positivos: 0.1<b<7, omega: 0.01<omega<0.43(2)
#Negativos: Sin problemas :D
