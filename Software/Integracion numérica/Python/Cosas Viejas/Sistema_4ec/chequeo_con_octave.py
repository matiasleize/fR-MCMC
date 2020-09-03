#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:38:48 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

#%% Importo los datos de Python H(z)
npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/H(z).npz')
zs_h = npzfile['zs']
hubbles = npzfile['hubbles']

# Importo los datos de Python d_L(z)
npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/d_L(z).npz')
zs_dl = npzfile['zs']
d_L = npzfile['d_L']

# Importo los datos de Octave H(z) y d_L
df = pd.read_csv(
        '/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/Datos_octave/datos_octave.txt'
                 , header = None, delim_whitespace=True)
#%%
# Comparo los H(z)
f = interp1d(np.array(df[0]),np.array(df[1]))
restas = (hubbles-np.array(f(zs_h)))/hubbles
plt.close()
plt.figure()
plt.plot(zs_h, restas,label=r'$\Gamma=\Gamma(y,v)$')

plt.xlabel('z(redshift)')
plt.ylabel(r'$\frac{H_{python}-H_{octave}}{H_{python}}$', size =20)
plt.legend(loc='best')
plt.grid(True)
#%%
plt.figure()
plt.plot(zs_h,hubbles)
plt.plot(zs_h,f(zs_h))

#%% Comparo los d_L(z)
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