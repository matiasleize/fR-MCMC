#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:48:22 2019

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps as simps
#CAMBIAR PATH
npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/H(z).npz')
zs = npzfile['zs']
hubbles = npzfile['hubbles']

#Calculamos la distancia luminosa d_L =  (c/H_0) int(dz'/E(z'))

from scipy.constants import c
H_0 =  74.2

d_c=np.zeros(len(hubbles)) #Distancia comovil

for i in range (1,len(hubbles)):
    d_c[i] = (c/H_0) * simps(1/hubbles[:i],zs[:i])

d_L = (1+zs) * d_c

#Guardamos la distancia luminosa


np.savez('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/d_L(z)', zs=zs, d_L=d_L)

#%%
plt.plot(zs,d_L, label= '$H_{0}=74.2$' )
#plt.plot(zs,mu, label='gamma = {}'.format(gamma))

plt.title('Distancia luminosa')
plt.xlabel('z(redshift)')
plt.ylabel(r'$d_L$')
plt.legend(loc='best')
plt.grid(True)
#%%
