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

from pc_path import definir_path
path_git, path_datos_global = definir_path()

os.chdir(path_git)
sys.path.append('./Software/Funcionales')
from funciones_int import integrador
from funciones_cambio_parametros import params_fisicos_to_modelo
os.chdir(path_git)
sys.path.append('./Software/Integracion numérica/Python/Sistema_4ec/')
#Importo los datos de Python H(z)
npzfile = np.load('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_5ec/H(z).npz')
zs_h_5 = npzfile['zs']
hubbles_5 = npzfile['hubbles']

#%%
# Comparo los H(z)
f = interp1d(zs_h_5,hubbles_5)
def error_H(f,H_comparado,z_comparado):
    H_comparador = f(z_comparado)
    eps_H = (H_comparador-H_comparado)/H_comparador
    return eps_H
#%%
#Coindiciones iniciales
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales


c_1 = 1
c_2 = 1/19 #(valor del paper)
C = 0.24
n = 1
params_modelo = [c_1,c_2,n] #de la cruz: [b,c,d,n]


%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)


#cantidad_zs = 1000
max_step = np.inf

#for max_step in np.linspace(0.01,0.1,3)#[0.1,0.05,0.01]:
for cantidad_zs in np.linspace(1000,2000,100):
    cantidad_zs = int(cantidad_zs)
    zs,hubble = integrador(ci, params_modelo,
                cantidad_zs=cantidad_zs, max_step=max_step)

    error =  error_H(f,hubble,zs)
    plt.plot(zs,error,'o',label='{}'.format(cantidad_zs))
    plt.legend(loc='best')
#%%
z_0 = 3

#Coindiciones iniciales
x_0 = -0.339
y_0 = 1.246
v_0 = 1.64
w_0 = 1 + x_0 + y_0 - v_0
r_0 = 41
ci = [x_0, y_0, v_0, w_0, r_0] #Condiciones iniciales


c_1 = 1
c_2 = 1/19 #(valor del paper)
C = 0.24
n = 1
params_modelo = [c1,C,c2,n] #de la cruz: [b,c,d,n]

%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)

N = 5
errores = np.zeros(N)
cantidades = np.zeros(N)
#cantidad_zs = 1000
max_step = 0.1
#for max_step in np.linspace(0.01,0.1,N)#[0.1,0.05,0.01]:
for i, cantidad_zs in enumerate(np.linspace(100,1000,N)):
    cantidad_zs = int(cantidad_zs)
    zs,hubble = integrador(ci, params_modelo,
                cantidad_zs=cantidad_zs, max_step=max_step)

    errores[i] =  error_H(f,hubble[-1],z_comparado=z_0)
    cantidades[i] =  cantidad_zs

plt.plot(cantidades,errores,'-.')
