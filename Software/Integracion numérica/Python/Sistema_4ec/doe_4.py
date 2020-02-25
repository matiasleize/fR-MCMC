#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:03:17 2019

@author: matias
"""

from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
import numpy as np
from matplotlib import pyplot as plt

#%%
def dX_dz(z, variables):

    x = variables[0]
    y = variables[1]
    v = variables[2]
    w = variables[3]

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*gamma(y,v) - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*gamma(y,v) + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x+ 2*v) / (z+1)

    return [s0,s1,s2,s3]

def plot_sol(solucion, gamma):

    '''Dado un gamma y una solución de las variables dinamicas, grafica estas
    por separado en una figura de 4x4.'''

    f, axes = plt.subplots(2,2)
    ax = axes.flatten()

    if isinstance(gamma, int):
        f.suptitle('$\Gamma$={}'.format(gamma), fontsize=15)
    else:
        f.suptitle('$\Gamma$=funcion'.format(gamma), fontsize=15)
    color = ['b','r','g','y']
    y_label = ['x','y','v','w']
    [ax[i].plot(solucion.t,solucion.y[i],color[i]) for i in range(4)]
    [ax[i].set_ylabel(y_label[i],fontsize='medium') for i in range(4)];
    [ax[i].set_xlabel('z (redshift)',fontsize='medium') for i in range(4)];
    [ax[i].invert_xaxis() for i in range(4)]; #Doy vuelta los ejes
    plt.show()

#%%
##Coindiciones iniciales e intervalo

x_0 = -0.339
y_0 = 1.246
v_0 = 1.64

w_0 = 1 + x_0 + y_0 - v_0

ci = [x_0, y_0, v_0, w_0] #Condiciones iniciales

zi = 0
#zf = 3 # Es un valor razonable con las SN 1A
zf = 3
#%% Integramos el vector v y calculamos el Hubble

#gamma_1 =  lambda y,v: y/v
#gamma_2 =  lambda y,v: y/(n*(y-v))
gamma =  lambda y,v: 0.5 * v*y/(v-y)**2 #Hu-Sawicki

sol = solve_ivp(dX_dz, [zi,zf], ci, max_step=0.1)
#plot_sol(sol,gamma)

# Guardamos z y v(z)
np.savez('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_4ec/v(z)'
         , zs=sol.t, v=sol.y[2])

#%%
zs = np.linspace(0,3,7000)
hubbles = np.zeros(len(zs))
#plt.close()

for i in range(len(zs)):
    zi = 0
    zf = zs[i]
    sol = solve_ivp(dX_dz, [zi,zf], ci, max_step=0.1)

    int_v = simps(sol.y[2]/(1+sol.t),sol.t)
    hubbles[i]=(1+zf)**2 * np.e**(-int_v) # integro desde 0 a z, ya arreglado


plt.plot(zs,hubbles,label=r'$\Gamma=\Gamma(y,v)$')
plt.title('Parámetro de Hubble')
plt.xlabel('z(redshift)')
plt.ylabel(r'$H(z)/H_{0}$')

plt.legend(loc='best')
plt.grid(True)

# Guardamos z y H()
np.savez('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_4ec/H(z)'
         , zs=zs, hubbles=hubbles)

#%% Chequeo
cte = sol.y[3]+sol.y[2]-sol.y[1]-sol.y[0]

plt.plot(sol.t,cte)
