#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:38:48 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import sys
import os
from os.path import join as osjoin
import time
from scipy.integrate import solve_ivp
from scipy.integrate import simps as simps

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales')
from funciones_int import plot_sol
from funciones_cambio_parametros import params_fisicos_to_modelo


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

    #nombre_modelo = params_modelo[-1]
    #if nombre_modelo == 'Star':
#        gamma = lambda r,b,c,d,n : b*r
#        [B,C,D,N,_] = params_modelo
#        G = gamma(r,B,C,D,N)
#    elif nombre_modelo == 'HS':
        #if params_modelo[3]==1:
        #    gamma = lambda y,v: y*v/(2*(y-v)**2)
        #    G = gamma(y,v)
        #else:
            #[b,d,c,n] = parametros_modelo
    gamma = lambda r,b,c,d,n: ((1+d*r**n) * (-b*n*r**n + r*(1+d*r**n)**2)) / (b*n*r**n * (1-n+d*(1+n)*r**n))
    [B,C,D,N] = params_modelo
    G = gamma(r,B,C,D,N)

    s0 = (-w + x**2 + (1+v)*x - 2*v + 4*y) / (z+1)
    s1 = - (v*x*G - x*y + 4*y - 2*y*v) / (z+1)
    s2 = -v * (x*G + 4 - 2*v) / (z+1)
    s3 = w * (-1 + x + 2*v) / (z+1)
    s4 = -x*r*G/(1+z)
    return [s0,s1,s2,s3,s4]

sistema_ec=dX_dz
z_inicial=0
z_final=3
cantidad_zs = 2000
max_step=0.01
verbose=True


x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales
cond_iniciales=ci

#c1_true = 1
#r_hs_true/H_0**2  = 0.24
#c2_true = 1/19
#n=1
params_modelo=[1,0.24,1/19,1]



#%% Forma nueva de integrar
plt.figure()
zs = np.linspace(z_inicial,z_final,cantidad_zs)
t1 = time.time()
sol = solve_ivp(sistema_ec, [z_inicial,z_final],
      cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)
t2 = time.time()
print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
      int((t2-t1) - 60*int((t2-t1)/60))))
plot_sol(sol)


E=np.ones(len(sol. t))
for i in range (1, len(sol.t)):
    int_v =  simps((sol.y[2][:i])/(1+sol.t[:i]),sol.t[:i])
    E[i] = (1+sol.t[i])**2 * np.e**(-int_v)
plt.plot(sol.t,E)


#%% Forma vieja de integrar
plt.figure()
t1 = time.time()
zs = np.linspace(z_inicial,z_final,cantidad_zs)
hubbles_1 = np.zeros(len(zs))
for i in range(len(zs)):
    zf = zs[i] # ''z final'' para cada paso de integracion
    sol_1 = solve_ivp(sistema_ec,[z_inicial,zf], cond_iniciales,args=params_modelo, max_step=max_step)
    int_v = simps((sol_1.y[2])/(1+sol_1.t),sol_1.t) # integro desde 0 a z
    hubbles_1[i] = (1+zf)**2 * np.e**(-int_v)
t2 = time.time()
print('Duración {} minutos y {} segundos'.format(int((t2-t1)/60),
      int((t2-t1) - 60*int((t2-t1)/60))))
plt.plot(zs,hubbles_1)
#%%
value = 1
plt.plot(zs[zs>value],(hubbles_1[zs>value]-E[zs>value])/E[zs>value])
#mientras mas grande z el error relativo cometido en la integracion es menor!
