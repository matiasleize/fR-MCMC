#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:48:22 2019

@author: matias
"""    

import numpy as np
from matplotlib import pyplot as plt

#%% Parámetro de desaceleración

npzfile = np.load('/home/matias/Documents/Tesis/Software/Integracion numérica/Python/Sistema_4ec/v(z).npz')
zs = npzfile['zs']
v = npzfile['v']

plt.plot(zs, 1-v, label= r'$\Gamma=\Gamma(y,v)$' )   
plt.title('Parámetro de desaceleración')
plt.xlabel('z(redshift)')
plt.ylabel(r'$q(z)$')
plt.legend(loc='best')
plt.grid(True)
