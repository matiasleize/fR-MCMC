import numpy as np
from matplotlib import pyplot as plt

import sys
import os
from os.path import join as osjoin
from pc_path import definir_path
path_git, path_datos_global = definir_path()
os.chdir(path_git)
sys.path.append('./Software/utils/')
from funciones_condiciones_iniciales import condiciones_iniciales
from funciones_cambio_parametros import params_fisicos_to_modelo
from funciones_int import integrador
from funciones_taylor import Taylor_HS
#%%

b = 0.01
omega_m = 0.24
H0 = 73.48
params_fisicos = [omega_m,b,H0]

zs, H_ode = integrador(params_fisicos)
H_taylor = Taylor_HS(zs,omega_m,b,H0)

#%%
#%matplotlib qt5
plt.close()
plt.figure()
plt.grid(True)
plt.xlabel('z(redshift)')
plt.ylabel('E(z)')
plt.plot(zs,H_taylor/H0,label='Taylor')
plt.plot(zs,H_ode/H0,label='ODE')
plt.legend(loc='best')

#%% Calculo el error porcentual entre la ODE y el taylor a b fijo.
plt.figure()
plt.grid(True)
plt.xlabel('z (redshift)')
plt.ylabel('Error porcentual')
plt.plot(zs,100*(1-(H_taylor/H_ode)),label='Error porcentual')
plt.legend(loc='best')
plt.show()
np.mean(100*(1-(H_taylor/H_ode)))
