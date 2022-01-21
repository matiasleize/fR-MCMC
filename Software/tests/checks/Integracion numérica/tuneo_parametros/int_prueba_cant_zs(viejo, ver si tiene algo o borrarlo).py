"""
Created on Sun Feb  2 13:28:48 2020

@author: matias
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import c as c_luz #metros/segundos
c_luz_norm = c_luz/1000

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from int import integrador


#%%

omega_m = 0.24
b = 0.2
H0 = 73.48
params_fisicos = [omega_m,b,H0]

z_inicial = 30
z_final = 0
cantidad_zs = int(10**6)

zs_patron  = np.linspace(z_inicial,z_final,10000)
cantidad_zs = np.linspace(10,10000,10)

cant_zs = np.array([int(i) for i in cantidad_zs])
print(cant_zs)
Hs = []

for i,cant in enumerate(cant_zs):

    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cant,
    max_step=0.003)

    f = interp1d(zs,H_ode)
    Hs.append(f(zs_patron))

final = []
for j in range(1,len(Hs)):
    aux = np.mean((Hs[j]-Hs[j-1])/Hs[j]);
    final.append(aux);
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('cantidad_zs')
plt.ylabel('%Hs')
plt.plot(cant_zs[1:],np.array(final)*100,'.');
#plt.legend(loc='best')
plt.show()
