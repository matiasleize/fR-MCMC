#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:38:48 2019

@author: matias

Comparo las soluciones de Python para dos gammas (validas) distintas
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
from scipy.integrate import cumtrapz as cumtrapz

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

    if (len(params_modelo)==3):
        [B,D,N] = params_modelo
        gamma = lambda r,b,d: ((1+d*r) * (r * (1+d*r)**2 - b*r)) / (2*b*d*r**2)
        G = gamma(r,B,D)
    else:
        gamma = lambda y,v: y*v/(2*(y-v)**2)
        G = gamma(y,v)

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
#c2_true = 1/19
#n=1
params_modelo=[1,1/19,1]
H0=73.48

#%% Forma nueva de integrar
zs = np.linspace(z_inicial,z_final,cantidad_zs)

#Gamma(y,v) --> solo funciona para c1 = 1
sol_1 = solve_ivp(sistema_ec, [z_inicial,z_final],
      cond_iniciales,t_eval=zs,args=[], max_step=max_step)
int_v_1 =  cumtrapz(sol_1.y[2]/(1+sol_1.t),sol_1.t,initial=0)
E1 = (1+sol_1.t)**2 * np.exp(-int_v_1)
H_1 = H0 * E1

#Gamma(r,c1,c2) --> la de Lucila
sol_2 = solve_ivp(sistema_ec, [z_inicial,z_final],
      cond_iniciales,t_eval=zs,args=params_modelo, max_step=max_step)
int_v_2 =  cumtrapz(sol_2.y[2]/(1+sol_2.t),sol_2.t,initial=0)
E2 = (1+sol_2.t)**2 * np.exp(-int_v_2)
H_2 = H0 * E2
#%%
plt.figure()
plt.plot(zs,(H_1-H_2)/H_1)
plt.grid(True)
