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
b = 0.1
H0 = 73.48
params_fisicos = [omega_m,b,H0]

cantidad_zs = int(10**5)

max_steps = np.linspace(0.01,0.001,100)
Hs = []
for maxs in max_steps:
    zs, H_ode = integrador(params_fisicos, n=1, cantidad_zs=cantidad_zs,
                max_step=maxs)
    #f=interp1d(zs,H_ode)
    #Hs.append(f(np.linspace(0,3,100000)))
    Hs.append(H_ode)
final = np.zeros(len(Hs))
for j in range(1,len(Hs)):
    aux = np.mean(Hs[j]-Hs[j-1]);
    #aux = np.mean((1-Hs[j]/Hs[j-1]));
    final[j]=aux;
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)

plt.xlabel('Tamaño del paso de integración', fontsize=13)
plt.ylabel('$\Delta$H', fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(max_steps[::-1],np.array(final)[::-1],'.-');
plt.gca().invert_xaxis()

#plt.legend(loc='best',prop={'size': 12})
plt.show()

#%%
index=np.where(abs(final)<=float(10**(-10)))[0][1]
max_steps[index+1]
final[index+1]

max_steps
final
#Delta H = 10**(-10)
#Delta h = 10**(-12/13)
