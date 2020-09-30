import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/Funcionales/')
from funciones_int import integrador

#%%
b = 2
omega_m = 0.24
H0 = 73.48
params_fisicos=[omega_m,b,H0]

#%%
archivo_math = '/Software/Sandbox/Integracion numérica/chequeos_ODE/chequeo_otros_lenguajes/Mathematica'
os.chdir(path_git+archivo_math)
z_math,E_math=np.loadtxt('mfile_ODE_{}.csv'.format(b),unpack=True,delimiter = ',')

zs, H_ode=integrador(params_fisicos,max_step=0.001,cantidad_zs=int(10**5))
f = interp1d(zs,H_ode)
H_int = f(z_math)
E_int = H_int/H0
#%%
%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('E(z)')
plt.plot(z_math,H_int/H0,label='Taylor')
plt.plot(z_math,E_math,label='Mathematica')
plt.legend(loc='best')

#%%
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('Error porcentual')
plt.plot(z_math,100*(1-E_int/E_math),label='Error porcentual')
plt.legend(loc='best')
np.mean(100*(1-(E_int/E_math)))
