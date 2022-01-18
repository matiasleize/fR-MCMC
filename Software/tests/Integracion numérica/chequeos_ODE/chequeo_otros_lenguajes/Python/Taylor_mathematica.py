import numpy as np
from matplotlib import pyplot as plt

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_taylor import Taylor_HS

#%%
z_inicial = 0
z_final = 3

b = 0.1
omega_m = 0.24
H0 = 73.48
params_fisicos=[omega_m,b,H0]

archivo_math = '/Software/Sandbox/Integracion numérica/chequeos_ODE/chequeo_otros_lenguajes/Mathematica'
os.chdir(path_git+archivo_math)
z_math,H_math=np.loadtxt('mfile_taylor_{}.csv'.format(b),unpack=True,delimiter = ',')

zs = np.linspace(z_inicial,z_final,len(math))
H_taylor=Taylor_HS(zs,omega,b,H0)

#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('E(z)')
plt.plot(zs,H_taylor/H0,label='Taylor')
plt.plot(z_math,H_math/H0,label='Mathematica')
plt.legend(loc='best')
#%%
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('Error porcentual')
plt.plot(zs,100*(1-H_taylor/H_math),label='Error porcentual')
plt.legend(loc='best')
np.mean(100*(1-(H_taylor/H_math)))
