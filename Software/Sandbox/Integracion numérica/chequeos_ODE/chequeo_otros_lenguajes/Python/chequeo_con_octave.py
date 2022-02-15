#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:38:48 2019

@author: matias

Comparo los resultados de la integracion en Python con los de Octave
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

import sys
import os
from os.path import join as osjoin

import time

from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps
from scipy.integrate import cumtrapz as cumtrapz

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales')
from funciones_int import plot_sol
from funciones_cambio_parametros import params_fisicos_to_modelo

# Importo los datos de Octave H(z) y d_L
df = pd.read_csv(
        '/home/matias/Documents/Tesis/fR-MCMC/Software/Integracion numérica/Python/analisis_int_solve_ivp/datos_octave.txt'
                 , header = None, delim_whitespace=True)



def dX_dz(z, variables,*params_modelo):
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
    gamma = lambda r,b,d: ((1+d*r) * ((1+d*r)**2 - b)) / (2*b*d*r)
    G = gamma(r,B,D)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x + 2*v) / (z+1)
    s4 = -(x*r*G)/(1+z)
    return [s0,s1,s2,s3,s4]

sistema_ec=dX_dz
z_inicial=0
z_final=3
cantidad_zs = 20000
max_step=0.01
max_step=np.inf
verbose=True


x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales=ci

#c1_true = 1
#c2_true = 1/19
#n=1
params_modelo=[1,1/19,1]
H0=73.48

#%% Forma nueva de integrar
zs = np.linspace(z_inicial,z_final,cantidad_zs)

sol_1 = solve_ivp(sistema_ec, [z_inicial,z_final],
      cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)
int_v_1 =  cumtrapz(sol_1.y[2]/(1+sol_1.t),sol_1.t,initial=0)
E1 = (1+sol_1.t)**2 * np.exp(-int_v_1)
H_1 = H0 * E1

if (len(zs)!=len(sol_1.t)):
    print('Está integrando mal!')

#%% Comparo los errores porcentuales para Python y Octave
f = interp1d(np.array(df[0]),np.array(df[1]))
restas = (1-(np.array(f(zs))/E1)) *100
plt.close()
plt.figure()
plt.plot(zs, restas,label=r'$\Gamma=\Gamma(C_{1},C_{2})$')
plt.xlabel('z (redshift)')
plt.ylabel(r'$\frac{H_{python}-H_{octave}}{H_{python}}\times $%$100$', size =20)
plt.legend(loc='best')
plt.grid(True)
#%% Comparo los E(z)
plt.figure()
plt.plot(zs,E1,label='Python')
plt.plot(zs,f(zs),'-.',label='Octave')
plt.xlabel('z (redshift)')
plt.ylabel('E(z)')
plt.legend(loc='best')
plt.grid(True)

#plt.plot(sol_1.t,sol_1.y[2])

#%% 17/07: Lo que sigue es viejo. Comparo los d_L(z)
g = interp1d(np.array(df[0]), np.array(df[2]))
restas_dl = (d_L-np.array(g(zs_dl)))/d_L
plt.close()
plt.figure()

plt.plot(zs_dl, restas_dl,label=r'$\Gamma=\Gamma(y,v)$')
plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{d_{python}-d_{octave}}{d_{python}}$, size =20)
plt.legend(loc='best')
plt.grid(True)

#%% No da tan bien!
plt.figure()
plt.plot(zs_dl,d_L)
plt.plot(zs_dl,g(zs_dl))
