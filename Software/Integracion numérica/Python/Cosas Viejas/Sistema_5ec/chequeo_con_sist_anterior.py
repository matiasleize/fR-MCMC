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
sys.path.append('./Software/Integracion numérica/Python/Sistema_4ec/')
#%% Version de 4 ec

#Importo los datos de Python H(z)
npzfile = np.load('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_4ec/H(z).npz')
zs_h_4 = npzfile['zs']
hubbles_4 = npzfile['hubbles']

# Importo los datos de Python d_L(z)
npzfile = np.load('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_4ec/d_L(z).npz')
zs_dl_4 = npzfile['zs']
d_L_4 = npzfile['d_L']

#%% Version de 5 ec

#Importo los datos de Python H(z)
npzfile = np.load('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_5ec/H(z).npz')
zs_h_5 = npzfile['zs']
hubbles_5 = npzfile['hubbles']

# Importo los datos de Python d_L(z)
npzfile = np.load('/home/matias/Documents/Tesis/tesis_licenciatura/Software/Integracion numérica/Python/Sistema_5ec/d_L(z).npz')
zs_dl_5 = npzfile['zs']
d_L_5 = npzfile['d_L']

#%%
# Comparo los H(z)
f = interp1d(zs_h_4,hubbles_4)
restas = (hubbles_5-np.array(f(zs_h_5)))/hubbles_5
plt.close()
plt.figure()
plt.plot(zs_h_5, restas)#,label=r'$\Gamma=\Gamma(y,v)$')

plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{H_{nuevo}-H_{viejo}}{H_{nuevo}}$', size =20)
#plt.legend(loc='best')
plt.grid(True)
#%%
plt.figure()
plt.plot(zs_h_5,hubbles_5)
plt.plot(zs_h_5,f(zs_h_5))

#%% Comparo los d_L(z)
g = interp1d(zs_dl_4, d_L_4)
restas_dl = (d_L_5-np.array(g(zs_dl_5)))/d_L_5
plt.close()
plt.figure()

plt.plot(zs_dl_5, restas_dl)#,label=r'$\Gamma=\Gamma(y,v)$')
plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{d_{nuevo}-d_{viejo}}{d_{nuevo}}$', size =20)
#plt.legend(loc='best')
plt.grid(True)

#%% Da bastante bien!
plt.figure()
plt.plot(zs_dl_5,d_L_5)
plt.plot(zs_dl_5,g(zs_dl_5))
